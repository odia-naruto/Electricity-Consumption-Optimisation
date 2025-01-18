import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
data2 = {}

def split_datetime(date):
    days=[0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    datetime = date.split(' ')
    date = datetime[0].split('-')
    return [days[int(date[1])-1]+int(date[2]),datetime[1]]

def kreg(data2):
    
    X = []
    y = []
    for i in range(len(data2)):
            X.append(i)
            y.append(data2[i])

    X = pd.DataFrame(X, columns=['week'])
    y = pd.Series(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    # Fit the kernel regression model with increased variance
    model = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    print(y_pred)
    print(y_test)
    print(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    # Plot the fitting curve along with test points and predicted points
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Test Points')
    plt.scatter(X_test, y_pred, color='blue', label='Predicted Points')
    plt.plot(X, model.predict(X), color='green', label='Fitting Curve')
    plt.xlabel('Week')
    plt.ylabel('Power')
    plt.title('Kernel Ridge Regression')
    plt.legend()
    plt.show()
def weekly_power(dataset, apt_num):
    global data2 
    # dataset = pd.read_csv('Apt2_2014.csv')
    dataset=dataset.values.tolist()
    data1=[]
    data = dict()
    data['power'] = []
    for i in range(53):
        data['power'].append(0)
    for i in dataset:
        temp=split_datetime(i[0])
        data['power'][int(temp[0]/7)]+=i[1]
    for i in data['power']:
        if i != 0:
            data1.append(i)
    data2[apt_num] = data1
    # print(len(data1))
    co=['blue','red','green']
    for i in data2:
        plt.plot(data2[i])
    print(data1)
    # plt.show()
    kreg(data1)
# testing datasets from apartment 2 4 and 5 for 2015
# d1=pd.read_csv('Apt5_2015.csv')
# # d2=pd.read_csv('Apt2_2015.csv')
# # d3=pd.read_csv('Apt4_2015.csv')
# weekly_power(d1,1)
# weekly_power(d2,2)
# weekly_power(d3,3)

# Function to fit regression model for multiple datasets and save the models
def fit_and_save_models(datasets, apt_nums):
    models = {}
    for dataset, apt_num in zip(datasets, apt_nums):
        weekly_power(dataset, apt_num)
        X = pd.DataFrame(range(len(data2[apt_num])), columns=['week'])
        y = pd.Series(data2[apt_num])
        model = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
        model.fit(X, y)
        models[apt_num] = model
        joblib.dump(model, f'model_apt_{apt_num}.pkl')
    return models

# # Example usage with 10 datasets
datasets = [pd.read_csv(f'Apt{i}_2015.csv') for i in [2,4,5]]
apt_nums = [2, 4, 5]
models = fit_and_save_models(datasets, apt_nums)

def predict_with_models(models):
    ip = int(input("Enter a day : "))
    ip = [ip]
    print(ip)
    predictions = []
    a = pd.DataFrame(ip, columns=['week'])
    for apt_num, model in models.items():
        predictions.append(model.predict(a)[0])
    return predictions

# # Example usage with new data
# new_data = {i: pd.read_csv(f'New_Apt{i}_2016.csv')['power'].tolist() for i in [2, 4, 5]}
predictions = predict_with_models(models)
print(predictions)
