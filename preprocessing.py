# Does the preprocessing which is done in ./notebooks/flickr100m/data_exploration.ipynb
import sys
import pandas as pd

data_dir = "/Users/davidmalmstrom/Documents/proj/data/flickr100m/"
try:
    nrows = int(sys.argv[1])
except IndexError:
    nrows = 5000000

print("Reading dataset...")
dataset = pd.read_csv(data_dir + "yfcc100m_dataset", sep="\t", nrows=nrows, header=None)
dataset.columns = pd.read_csv(data_dir + "dataset_header").columns
dataset['user+id'] = dataset['User_NSID'].map(str) + "/" + dataset['Photo/video_identifier'].map(str)
dataset.set_index('user+id', inplace=True)

print("Reading autotags...")
autotags = pd.read_csv(data_dir + "yfcc100m_autotags", sep="\t", nrows=nrows, header=None)
autotags = autotags.dropna()

print("")
# remove confidence values
tag_lists = autotags[1].map(lambda x: list(map(lambda y: y.split(':')[0], str(x).split(','))))
autotags['autotags'] = tag_lists
autotags = autotags.drop(1, axis=1)

autotags.set_index(0, inplace=True)
dataset = dataset.join(autotags, on='Photo/video_identifier')

# User tags to array:
dataset.User_tags = dataset.User_tags.map(lambda x: x.split(','), na_action='ignore')

# Some stats printing:
all_autotags = pd.Series((tag for tag_list in dataset.autotags.dropna() for tag in tag_list))
print("number of autotags:")
print("total: " + str(len(all_autotags)))
print("unique: " + str(len(all_autotags.unique())))
print("ratio autotags/item: " + str(len(all_autotags) / dataset.shape[0]))
print("")
all_usertags = pd.Series((tag for tag_list in dataset.User_tags.dropna() for tag in tag_list))
print("number of user tags:")
print("total: " + str(len(all_usertags)))
print("unique: " + str(len(all_usertags.unique())))
print("ratio user_tags/item: " + str(len(all_usertags) / dataset.shape[0]))


prepare_dataset = dataset[['User_tags', 'autotags']]
prepare_dataset.head(2)
prepare_dataset = prepare_dataset.dropna()

prepare_dataset = prepare_dataset.query('User_tags.str.len() > 5')

all_user_tags = pd.Series((tag for tag_list in prepare_dataset.User_tags for tag in tag_list))

user_value_counts = all_user_tags.value_counts()
common_user_tags = set(user_value_counts[user_value_counts > 110].index)

prepare_dataset.User_tags = prepare_dataset.User_tags.map(lambda tag_list: [tag for tag in tag_list if tag in common_user_tags])

prepare_dataset = prepare_dataset.query('User_tags.str.len() > 5')

all_asd = pd.Series((tag for tag_list in prepare_dataset.User_tags for tag in tag_list))

# more prints:
auto_tag_list = pd.Series((tag for tag_list in prepare_dataset.autotags for tag in tag_list))
user_tag_list = pd.Series((tag for tag_list in prepare_dataset.User_tags for tag in tag_list))
print("The average amount of autotags per item is " + str(len(auto_tag_list)/prepare_dataset.shape[0]) + ".")
print("number of autotags:")
print("total: " + str(len(auto_tag_list)))
print("unique: " + str(len(auto_tag_list.unique())))
print("ratio autotags/item: " + str(len(auto_tag_list) / prepare_dataset.shape[0]))
print("")
print("number of user tags:")
print("total: " + str(len(user_tag_list)))
print("unique: " + str(len(user_tag_list.unique())))
print("ratio user_tags/item: " + str(len(user_tag_list) / prepare_dataset.shape[0]))

print("Size of final dataset: " + str(prepare_dataset.shape[0]))

prepare_dataset.to_pickle("results/preprocessed_user_auto_tags_big.pkl")