import mysql.connector
import streamlit as st
import plotly.express as px
import pandas as pd

# Establish connection to MySQL database
def fetch_data():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="shop3"
        )
        cursor = connection.cursor()

        # Query to fetch product, category, and count columns
        query = "SELECT product, category, count FROM sold"
        cursor.execute(query)
        product_results = cursor.fetchall()

        cursor.close()
        connection.close()

        return product_results

    except mysql.connector.Error as err:
        st.error(f"Database Error: {err}")
        return []
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        return []

# Fetch data
product_results = fetch_data()

if product_results:
    # Convert data to a DataFrame for better handling
    df = pd.DataFrame(product_results, columns=['Product', 'Category', 'Count'])

    # Create an interactive bar chart using Plotly
    st.subheader("Interactive Product Count Bar Chart")
    bar_chart = px.bar(
        df,
        x='Product',
        y='Count',
        title='Product Count in Sold Table',
        labels={'Product': 'Product Names', 'Count': 'Count'},
        color='Count',
        text='Count'
    )
    bar_chart.update_traces(texttemplate='%{text}', textposition='outside')
    bar_chart.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(bar_chart)

    # Aggregate data by category for the pie chart
    category_data = df.groupby('Category', as_index=False).sum()

    # Create an interactive pie chart using Plotly
    st.subheader("Interactive Category Distribution Pie Chart")
    pie_chart = px.pie(
        category_data,
        names='Category',
        values='Count',
        title='Category Distribution in Sold Table',
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    pie_chart.update_traces(textinfo='percent+label')
    st.plotly_chart(pie_chart)
else:
    st.warning("No data available to display charts.")
