import cv2
import os
import pickle
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
MATCH_TRESHOLD = 160

def extract_and_save_features(base_directory, detector):
    # Iterate through all folders in the base directory
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        
        if os.path.isdir(folder_path):
            # Process only the TCI images in each folder
            for image_file in glob.glob(os.path.join(folder_path, '*_TCI.jpg')):
                # Extract features
                img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                keypoints, descriptors = detector.detectAndCompute(img, None)

                # Convert keypoints to a serializable format
                keypoints_serializable = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

                # Save keypoints and descriptors in the same folder
                keypoints_file = os.path.join(folder_path, 'keypoints.pkl')
                descriptors_file = os.path.join(folder_path, 'features.pkl')

                with open(keypoints_file, 'wb') as kp_file:
                    pickle.dump(keypoints_serializable, kp_file)
                with open(descriptors_file, 'wb') as des_file:
                    pickle.dump(descriptors, des_file)

def load_features(directory, image_features):
    # Iterate through all subdirectories in the directory
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            # Load descriptors from each folder
            descriptor_file = os.path.join(folder_path, 'features.pkl')
            if os.path.exists(descriptor_file):
                with open(descriptor_file, 'rb') as f:
                    descriptors = pickle.load(f)
                # The image file name is derived from the folder name
                image_file = glob.glob(os.path.join(folder_path, '*.jpg'))[0]  
                image_features[image_file] = descriptors
    return image_features

def compare_and_record_matches(args):
    img_path1, img_path2, des1, des2, threshold = args
    num_matches = compare_features(des1, des2, threshold)
    if num_matches > MATCH_TRESHOLD:
        return (img_path1, img_path2)
    return None

#threshold=0.75 will filter good matches between images
def compare_features(des1, des2, threshold=0.75):
    bf = cv2.BFMatcher() # Initialize Brute-Force Matcher 
    matches = bf.knnMatch(des1, des2, k=2) # match des1 and des2 with k=2 nearest neighbors.
    good_matches = [m for m, n in matches if m.distance < threshold * n.distance] # Apply ratio test and filtrer good matches
    return len(good_matches)

def preprocess_and_save_images(base_dataset_path, dev_dataset_path, max_dimension=1024):
    if not os.path.exists(dev_dataset_path):
        print(f"crerated {dev_dataset_path} to contain pictures and keypoints")
        os.makedirs(dev_dataset_path)

    for folder in os.listdir(base_dataset_path):
        # Construct the path to the IMG_DATA folder where TCI images are stored
        img_data_path = os.path.join(base_dataset_path, folder, "*.SAFE/GRANULE/*/IMG_DATA/*_TCI.jp2")
        
        # Find the TCI image file
        tci_image_file = glob.glob(img_data_path)
        
        for image_file in tci_image_file:
            
            subdirectory_name = image_file.split('/')[-3]
            dest_subdirectory_path = os.path.join(dev_dataset_path, subdirectory_name)
            
            if not os.path.exists(dest_subdirectory_path):
                os.makedirs(dest_subdirectory_path)
                
            # Preprocess the image
            preprocessed_image = load_and_preprocess_image(image_file, max_dimension)
            
            # Construct the path for saving the preprocessed image
            dest_image_path = os.path.join(dest_subdirectory_path, os.path.basename(image_file).replace('.jp2', '.jpg'))
            # Save the preprocessed image
            cv2.imwrite(dest_image_path, preprocessed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

def load_and_preprocess_image(image_path, max_dimension=1024):
    img = cv2.imread(image_path)
    height, width = img.shape[0], img.shape[1]
    scale = max_dimension / max(height, width)
    resized_img = cv2.resize(img, (int(width * scale), int(height * scale)))
    return resized_img

def save_matched_images(image_path1, image_path2, detector, directory, name):
    # Load images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    # turn them to gray
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # Draw first 20 matches
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
    cv2.imwrite(f"{directory}/{name}.jpg",result)

def create_matches():
    # downloaded data
    base_dataset_path = './data/'
    dev_dataset_path = './dev_data/'
    if os.path.exists(dev_dataset_path):
        if any(os.scandir(dev_dataset_path)):
            print(f"You have not empty folder {dev_dataset_path}, so it's probably contains our images in .jpg format")
            print("We will skip step of resizing and converting images")
            print(f"Note: if you have new images in {base_dataset_path} but old in dev_data - delete {dev_dataset_path} folder and re-run inference")
        else:
            # directory exists, but empty
            preprocess_and_save_images(base_dataset_path, dev_dataset_path)
    else:
        # directory doesn't exists
        preprocess_and_save_images(base_dataset_path, dev_dataset_path)
    print(f"All resized and preproccesed photos was saved in {dev_dataset_path}")
    # create a detector
    sift = cv2.SIFT_create()
    extract_and_save_features(dev_dataset_path, sift)
    # create a dict with filename as key and features as values
    image_features = {}
    image_features = load_features(dev_dataset_path, image_features)
    image_paths = list(image_features.keys()) 
    # a list with files to compare
    compare_args = []
    for i, img_path1 in enumerate(image_paths):
        for img_path2 in image_paths[i+1:]:
            compare_args.append((img_path1, img_path2, image_features[img_path1], image_features[img_path2], 0.75))
    #Comparing images
    total_comparisons = 0
    match_count = 0
    matched_image_paths = []
    print("starting finding matching images...")
    with ProcessPoolExecutor() as executor:
        future_to_comparison = {executor.submit(compare_and_record_matches, args): args for args in compare_args}
        for future in as_completed(future_to_comparison):
            result = future.result()
            total_comparisons += 1
            if result:
                match_count += 1
                matched_image_paths.append(result)

    print(f"Total number of matches found: {match_count}")
    print(f"Total number of comparisons made: {total_comparisons}") 
    directory_name = 'matched_results'
    # Create the directory
    os.makedirs(directory_name, exist_ok=True)
    print(f"Directory '{directory_name}' will contain results.")
    for index, matched in enumerate(matched_image_paths):
        save_matched_images(matched[0], matched[1], sift, directory_name, index)
    print("Matching completed.")


if __name__ == "__main__":
    create_matches()

