from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import random
from pymongo import MongoClient
from datetime import datetime

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
            print "File " + uploaded_file.filename + " uploaded successfully"

            # Load the CSV file into a Pandas DataFrame
            df = pd.read_csv(uploaded_file.filename,encoding='utf-8')          

            # Clean the data by removing non-ascii characters, missing values, duplicates, and outliers
            df.dropna(inplace=True)  # Remove missing values
            df.drop_duplicates(inplace=True)  # Remove duplicates
            
            # Remove outliers using a simple Z-score approach
            z_scores = (df['Quantity'] - df['Quantity'].mean()) / df['Quantity'].std()
            df = df[(z_scores < 3) & (z_scores > -3)]

            # Generate insights and visualizations
            # Example 1: Calculate total sales by month and plot as a bar chart
            df['Date'] = pd.to_datetime(df['Purchase_date'], format='%m/%d/%Y')
            df['Month'] = df['Date'].dt.month
            monthly_sales = df.groupby('Month')['Quantity'].sum()
            fig1, ax1 = plt.subplots()
            monthly_sales.plot(kind='bar', title='Total Sales by Month', ax=ax1)
            fig1.savefig('static/bar_chart.png')

            # Example 2: Calculate average sales by product category and plot as a pie chart
            category_sales = df.groupby('Product_category')['Quantity'].mean()
            fig2, ax2 = plt.subplots()
            category_sales.plot(kind='pie', title='Average Sales by Category', ax=ax2)
            fig2.savefig('static/pie_chart.png')

            # Pass the data to the display page
            columns = df.columns.tolist()
            data = df.values.tolist()
            bar_chart_path = 'static/bar_chart.png'
            pie_chart_path = 'static/pie_chart.png'

            return render_template('display.html', columns=columns, data=data, bar_chart_path=bar_chart_path, pie_chart_path=pie_chart_path)

    return 'Error: no file uploaded'

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
    app.run(debug=True, port=5000)
