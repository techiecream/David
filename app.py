from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

app = Flask(__name__)

# Create a MongoDB client instance
client = MongoClient('mongodb://localhost:27017/')

# Connect to the database
db = client['product_database']

# Get the collection
dbproducts = db['products']

# Delete the collection
#dbproducts.drop()



@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload')
def upload_file():
    return render_template('upload.html')
@app.route('/manual_entry')
def manual_entry():
    return render_template('manual_entry.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(uploaded_file.filename)
            print("File " + uploaded_file.filename + " uploaded successfully")
            # Load the CSV file into a Pandas DataFrame
            df = pd.read_csv(uploaded_file.filename, encoding='utf-8')

            #df = pd.read_csv('data.csv')
            df.head()

            df.describe()

            # datatype info
            df.info()

            # find unique values
            df.apply(lambda x: len(x.unique()))

            # distplot for purchase
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(13, 7))
            sns.distplot(df['Quantity'], bins=25)
            plt.savefig('distplot.png')
            

            # distribution of numeric variables
            sns.countplot(df['Product_name'])
            plt.savefig('product_count.png')

            sns.countplot(df['Quantity'])
            plt.savefig('quantity_count.png')
            

            sns.countplot(df['Sales_per_week'])
            plt.savefig('sales_person_count.png')

            sns.countplot(df['Sales_person'])

            sns.countplot(df['Purchase_date'])

            # check for null values
            df.isnull().sum()

            # to improve the metric use one hot encoding
            # label encoding
            cols = ['Product_category', 'City', 'Sales_person']
            le = LabelEncoder()
            for col in cols:
                df[col] = le.fit_transform(df[col])
            df.head()

            corr = df.corr()
            plt.figure(figsize=(14, 7))
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            df.head()

            X = df.drop(columns=['Product_id','Product_name', 'Product_category', 'Purchase_date','City_code'])
            y = df['Quantity']

            def train(model, X, y):
                # train-test split
                x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
                model.fit(x_train, y_train)

                # predict the results
                pred = model.predict(x_test)

                # cross validation
                cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
                cv_score = np.abs(np.mean(cv_score))

                print("Results")
                print("MSE:", np.sqrt(mean_squared_error(y_test, pred)))
                print("CV Score:", np.sqrt(cv_score))

            model = RandomForestRegressor(n_jobs=-1)
            train(model, X, y)

            features = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
            features.plot(kind='bar', title='Feature Importance')

            # Define x_test before making predictions
            x_test = df.drop(columns=['Product_id','Product_name', 'Product_category', 'Purchase_date','City_code'])
            pred = model.predict(x_test)

            # Add predicted sales to the existing 'Sales_per_week' column
            df['Predicated_sales'] = df['Sales_per_week'] + pred

            # Save the updated DataFrame to a CSV file
            df.to_csv('submission.csv', index=False)



        # Return the error message if no file is uploaded
        return render_template('upload.html', error='Error: no file uploaded')

    # Return the upload page for GET requests
    return render_template('upload.html')



def clean_and_process_data(df):
    df.head()

    df.describe()

    # datatype info
    df.info()

    # find unique values
    df.apply(lambda x: len(x.unique()))

    # distplot for purchase
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(13, 7))
    sns.distplot(df['Quantity'], bins=25)

    # distribution of numeric variables
    sns.countplot(df['Product_name'])

    sns.countplot(df['Quantity'])

    sns.countplot(df['Sales_per_week'])

    sns.countplot(df['Sales_person'])

    sns.countplot(df['Purchase_date'])

    # bivariate analysis
    occupation_plot = df.pivot_table(index='Sales_person', values='Quantity', aggfunc=np.mean)
    occupation_plot.plot(kind='bar', figsize=(13, 7))
    plt.xlabel('Sales_person')
    plt.ylabel("Quantity")
    plt.title("Sales_person and Quantity Analysis")
    plt.xticks(rotation=0)
    plt.show()

    # check for null values
    df.isnull().sum()

    # to improve the metric use one hot encoding
    # label encoding
    cols = ['Product_category', 'City', 'Sales_person']
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col])
    df.head()

    corr = df.corr()
    plt.figure(figsize=(14, 7))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    df.head()

    return df

def perform_analysis(df):
    X = df.drop(columns=['Product_id','Product_name', 'Product_category', 'Purchase_date','City_code'])
    y = df['Quantity']

    def train(model, X, y):
        # train-test split
        x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
        model.fit(x_train, y_train)

        # predict the results
        pred = model.predict(x_test)

        # cross validation
        cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
        cv_score = np.abs(np.mean(cv_score))

        print("Results")
        print("MSE:", np.sqrt(mean_squared_error(y_test, pred)))
        print("CV Score:", np.sqrt(cv_score))

    model = RandomForestRegressor(n_jobs=-1)
    train(model, X, y)

    features = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
    features.plot(kind='bar', title='Feature Importance')

    # Define x_test before making predictions
    x_test = df.drop(columns=['Product_id','Product_name', 'Product_category', 'Purchase_date','City_code'])
    pred = model.predict(x_test)

    submission = pd.DataFrame()
    submission['Product_id'] = df['Product_id']
    submission['Sales_per_week'] = pred
    submission.to_csv('submission.csv', index=False)




def generate_product_id():
    id_prefix = str(random.randint(1000, 9999))
    id_suffix = str(random.randint(1000, 9999))
    return "%s-%s" % (id_prefix, id_suffix)

@app.route('/submit', methods=['POST'])
def submit():
    # Get the data from the form
    product_id = generate_product_id()
    product_name = request.form['product_name']
    product_category = request.form['product_category']
    quantity = request.form['quantity']
    unit_price = request.form['unit_price']
    purchase_date = datetime.today().strftime('%m/%d/%Y')

    # Store the data in the MongoDB database
    product_data = {
        'product_id': product_id,
        'product_name': product_name,
        'product_category': product_category,
        'quantity': quantity,
        'unit_price': unit_price,
        'purchase_date': purchase_date
    }
    dbproducts.insert_one(product_data)

    return 'Data submitted successfully!'
@app.route('/liproducts')
def products():
    # Query the database for all products
    product_data = dbproducts.find({})
    product_list=[]
    for product in list(product_data):
        product_list.append(product)

    # Render the template with the product data
    return render_template('liproducts.html',  product_list=product_list)

if __name__ == '__main__':
    app.run(debug=True ,port=5000)
