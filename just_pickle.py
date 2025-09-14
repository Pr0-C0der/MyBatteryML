import pickle

# Replace 'your_file.pkl' with the path to your pickle file
file_path = "data\processed\HUST\HUST_1-1.pkl"

# Open the pickle file and load its content
with open(file_path, "rb") as file:  # "rb" means read in binary mode
    data = pickle.load(file)

# Print or use the data
print(type(data))
print(data)