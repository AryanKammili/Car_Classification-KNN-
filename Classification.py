import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

# Bring in our Data and Read it #
data = pd.read_csv("car.data")

print(data.head())

# Our Encoder is what allows us to transform String data to Integer Data #
labelEncoder = preprocessing.LabelEncoder()

# Transforming String Data to Integers ( 0 -3 ) #
buying = labelEncoder.fit_transform(list(data["buying"]))
maintenance = labelEncoder.fit_transform(list(data["maint"]))
door = labelEncoder.fit_transform(list(data["door"]))
persons = labelEncoder.fit_transform(list(data["persons"]))
lug_boot = labelEncoder.fit_transform(list(data["lug_boot"]))
safety = labelEncoder.fit_transform(list(data["safety"]))
cls = labelEncoder.fit_transform(list(data["class"]))

# Our Prediction #
predict = "class"

# Turns each one into a list with the data we wanted #
x = list(zip(buying, maintenance, door, persons, lug_boot, safety))
y = list(cls)

# Split up data for testing and training #
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Our number of neighbors can change, but should always be odd so there is winner #
model = KNeighborsClassifier(n_neighbors=9)

# Here we fit, find the accuracy, and predict the data #
# Highest accuracy was 94.5& #
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)

predictions = model.predict(x_test)

# These were the String data before it was turned to integers #
names = ["unacc", "acc", "good", "vgood"]

# Since it was 0 - 3 we can just watch the index with the names to get our string data #
for x in range(len(x_test)):
    print("Predicted: ", names[predictions[x]], " Data: ", x_test[x], "Actual: ", names[y_test[x]])






