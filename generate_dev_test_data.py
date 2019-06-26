import lib.notebook_helpers as nh
from sklearn.model_selection import train_test_split

path = "Data/"

dataset = nh.generate_data(20000, data_dir=path, data_name="preprocessed_user_auto_tags_big.pkl")

dev_dataset, test_dataset = train_test_split(dataset, test_size=0.1)

dev_dataset.to_pickle(path + "dev_tag_dataset.pkl")
test_dataset.to_pickle(path + "test_tag_dataset.pkl")