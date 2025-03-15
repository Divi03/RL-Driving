import pickle

def load_and_inspect_pickle(file_path):
    # Load the data from the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Check if the data is a list
    if isinstance(data, list):
        print(f"The pickle file contains a list with {len(data)} entries:")
        
        # Inspect the first few entries
        for index, entry in enumerate(data[:5]):  # Show first 5 entries for brevity
            print(f"Entry {index}: {entry}")
            print(f"  Type: {type(entry)}")  # Print the type of the tuple
            if isinstance(entry, tuple):
                print(f"  Size: {len(entry)}")  # Number of elements in the tuple
                for i, elem in enumerate(entry):
                    print(f"    Element {i} type: {type(elem)}")
                    if isinstance(elem, np.ndarray):  # If it's an ndarray, print its shape
                        print(f"    Element {i} shape: {elem.shape}")
                    elif isinstance(elem, (list, dict)):
                        print(f"    Element {i} length: {len(elem)}")  # Length for lists/dicts
            print()  # New line for better readability
            
    else:
        print("The data structure is neither a list.")

# Specify the path to your pickle file
pickle_file_path = '/Applications/Files/SEM_7/MAJOR/common/data/user_data_discrete_random_domain.pkl'  # Change to your actual path
load_and_inspect_pickle(pickle_file_path)
