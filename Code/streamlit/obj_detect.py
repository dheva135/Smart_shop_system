import numpy as np
import pandas as pd
import pyautogui
from ultralytics import YOLO
import cv2
import cvzone
import math
from PIL import Image
import streamlit as st
from obj_detect_img_video import *
#from drive import *
import tempfile
import mysql.connector
import plotly.express as px
import networkx as nx
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.offsetbox as offsetbox



# Generate a larger mall layout with ordered racks

def generate_large_mall_layout():
    layout = np.ones((20, 25), dtype=int)

    # Add obstacles (0s) with advertisements
    obstacles = [
        (5, 5), (5, 6), (5, 7),
        (15, 10), (16, 10), (17, 10),
        #(12, 20), (13, 20), (14, 20)
    ]
    for x, y in obstacles:
        layout[x, y] = 0

    # Add racks (2s) in an ordered manner
    racks = []
    for i in range(3, 20, 5):  # Rows spaced by 6
        for j in range(3, 25, 8):  # Two racks per row, spaced by 12 columns
            rack = [(i, j + k) for k in range(5)]  # 5 consecutive cells
            racks.append(rack)
            for x, y in rack:
                layout[x, y] = 2

    return layout, racks

# Create graph from the mall layout
def create_graph_from_layout(layout):
    G = nx.DiGraph()
    rows, cols = layout.shape

    # Define movement directions (4-way movement)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for x in range(rows):
        for y in range(cols):
            if layout[x, y] == 1:  # Only passable cells
                for dx, dy in directions:
                    neighbor_x, neighbor_y = x + dx, y + dy
                    if (
                        0 <= neighbor_x < rows
                        and 0 <= neighbor_y < cols
                        and layout[neighbor_x, neighbor_y] == 1
                    ):
                        G.add_edge((x, y), (neighbor_x, neighbor_y), weight=1)

    return G

# Check if shelf is accessible
def is_shelf_accessible(graph, shelf):
    x, y = shelf
    neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    return any(neighbor in graph.nodes for neighbor in neighbors)

# Find shortest path to any valid adjacent cell of the product shelf
def find_shortest_path_to_product(graph, start, product_shelf):
    x, y = product_shelf
    neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    accessible_neighbors = [node for node in neighbors if node in graph.nodes]

    shortest_path = None
    for neighbor in accessible_neighbors:
        try:
            path = nx.shortest_path(graph, source=start, target=neighbor, weight="weight")
            if shortest_path is None or len(path) < len(shortest_path):
                shortest_path = path
        except nx.NetworkXNoPath:
            continue

    return shortest_path

# Visualize the layout with paths and images
def visualize_large_layout(layout, path, selected_shelves, cart_image_path, rack_image_path, beacon_image_path, advert_image_path, exit_image_path):
    fig, ax = plt.subplots(figsize=(15, 15))

    ax.set_xlim(-0.5, layout.shape[1] - 0.5)
    ax.set_ylim(layout.shape[0] - 0.5, -0.5)
    ax.set_xticks(range(layout.shape[1]))
    ax.set_yticks(range(layout.shape[0]))
    ax.grid(which='major', color='black', linestyle='-', linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Load and resize images
    cart_image = Image.open(cart_image_path).resize((40, 40))
    rack_image = Image.open(rack_image_path).resize((40, 40))
    beacon_image = Image.open(beacon_image_path).resize((40, 40))
    advert_image = Image.open(advert_image_path).resize((40, 40))
    exit_image = Image.open(exit_image_path).resize((40, 40))

    # Plot the grid
    for x in range(layout.shape[0]):
        for y in range(layout.shape[1]):
            if layout[x, y] == 2:
                ab = offsetbox.AnnotationBbox(offsetbox.OffsetImage(rack_image), (y, x), frameon=False)
                ax.add_artist(ab)
            elif layout[x, y] == 0:
                ab = offsetbox.AnnotationBbox(offsetbox.OffsetImage(advert_image), (y, x), frameon=False)
                ax.add_artist(ab)

    # Highlight product shelves with beacons
    for product, (x, y) in selected_shelves.items():
        ab = offsetbox.AnnotationBbox(offsetbox.OffsetImage(beacon_image), (y, x), frameon=False, xybox=(0, 0.5), boxcoords="offset points")
        ax.add_artist(ab)
        ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color="lightgreen", alpha=0.7))
        ax.text(y, x, f"{product.capitalize()}", color='black', ha='center', va='center', fontsize=10, fontweight='bold')

    # Plot the cart image at the starting point
    start_x, start_y = path[0]
    ab = offsetbox.AnnotationBbox(offsetbox.OffsetImage(cart_image), (start_y, start_x), frameon=False)
    ax.add_artist(ab)

    # Draw the path and highlight adjacent cells
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        ax.arrow(y1, x1, y2 - y1, x2 - x1, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        if any((x2, y2) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] for x, y in selected_shelves.values()):

            ax.add_patch(plt.Rectangle((y2 - 0.5, x2 - 0.5), 1, 1, color="lightgreen", alpha=0.7))

    # Add the exit image
    ab = offsetbox.AnnotationBbox(offsetbox.OffsetImage(exit_image), (layout.shape[1] - 1, layout.shape[0] - 1), frameon=False)
    ax.add_artist(ab)

    plt.savefig("mall_path_with_images.png")
    plt.close(fig)

def main():

    #st.markdown("<center>welcome</center>",unsafe_allow_html=True,)

    #-------------HOME PAGE---------------------------------

    st.header('WELCOME TO e-SHOPPING CART', divider='rainbow')
    #st.markdown("</center>",unsafe_allow_html=True)
    st.title("_SHOP XYZ_")
    st.sidebar.header("MENU",divider='rainbow')
    #st.sidebar.subheader("Parameters")

    app_mode = st.sidebar.selectbox('Choose the Mode', ['HOME','LOGIN','PRODUCTS','ANALYSIS'])

    if app_mode == 'HOME':
        st.markdown(
            ':blue[This project uses **YOLO** for Object Detection on Images and Videos and we are using **StreamLit** to create a Graphical User Interface (GUI)]')
        st.image(
            'https://play-lh.googleusercontent.com/z2pE7U4gpS3A4QKDMaMGqJTHFcQ_-rZMkjQ7IHYJk2gHONJg1xQJP-HAwGwBLbE1Exs')

        # st.title("Enter your details")
        # user_phno = st.text_input("Mobile Number")
        # if st.button("Submit"):
        #     st.success("Continue your Shopping... navigate to shopping cart in sidebar")

    elif app_mode == 'LOGIN':
        st.sidebar.markdown('---')
        st.sidebar.header('Enter the Details')
        usr_name = st.sidebar.text_input('User name')
        phno = st.sidebar.text_input('Mobile Number')
        if st.sidebar.checkbox("Click Me !!!"):

                st.sidebar.markdown('---')
                #use_webcam = st.sidebar.checkbox('Use Webcam')
                st.sidebar.info("Upload the video")
                st.sidebar.markdown('---')
                st.header(f"Welcome {usr_name}, Continue your shopping !!!")
                uploaded_file_top = st.file_uploader("Upload a TOP-view video", type=["mp4", "avi"])
                uploaded_file_side = st.file_uploader("Upload a SIDE-view video", type=["mp4", "avi"])

                if uploaded_file_top and uploaded_file_side is not None:
                  # Save the uploaded file to a temporary location
                    st.sidebar.success("Running Video")
                    with open("temp_video1.mp4", "wb") as temp_file_top:
                        temp_file_top.write(uploaded_file_top.read())

                    with open("temp_video2.mp4", "wb") as temp_file_side:
                        temp_file_side.write(uploaded_file_side.read())


                stframe_s = st.empty()
                stframe_t = st.empty()

                #------------- DISPLAY DETAILS ------------------------------

                st.markdown("<hr/>", unsafe_allow_html = True)
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                with kpi1:
                    st.markdown("**Current ToTal**")
                    kpi1_text = st.markdown("<h1 style='color:black;'>0</h1>",unsafe_allow_html=True)
                    #kpi1_text.write(f"<h1 style='text-align:center; color:red;'>0</h1>")
                with kpi2:
                    st.markdown("**Amount**")
                    #kpi2_text = st.markdown("0")
                    kpi2_text = st.markdown("<h1 style='color:red;'>0</h1>",unsafe_allow_html=True)
                with kpi3:
                    st.markdown("**Stock**")
                    #kpi3_text = st.markdown("0")
                    kpi3_text = st.markdown("<h1 style='color:red;'>0</h1>",unsafe_allow_html=True)
                with kpi4:
                    st.markdown("**Product**")
                    # kpi3_text = st.markdown("0")
                    kpi4_text = st.markdown("<h1 style='color:red;'>0</h1>", unsafe_allow_html=True)


                st.markdown("<hr/>", unsafe_allow_html = True)
                st.markdown("<hr/>", unsafe_allow_html = True)
                kpi5,kpi6=st.columns(2)

                with kpi5:
                    #st.markdown("**reciept**")
                    kpi5_text = st.markdown("<h1 style='color:white;>0</h1>", unsafe_allow_html=True)

                st.markdown("<hr/>", unsafe_allow_html=True)

                if uploaded_file_top and uploaded_file_side is not None:
                    load_product_counter(temp_file_side.name,temp_file_top.name, kpi1_text,kpi2_text,  kpi3_text, kpi4_text,kpi5_text,stframe_s,stframe_t,usr_name,phno)

        else:
            st.markdown(
                ':blue[This project uses **YOLO** for Object Detection on Images and Videos and we are using **StreamLit** to create a Graphical User Interface (GUI)]')
            st.image(
                'https://play-lh.googleusercontent.com/z2pE7U4gpS3A4QKDMaMGqJTHFcQ_-rZMkjQ7IHYJk2gHONJg1xQJP-HAwGwBLbE1Exs')

    elif app_mode=="ANALYSIS":
        connection = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="",
                    database="shop3"
                )
        cursor = connection.cursor()

        # Query to fetch product and count columns
        query = "SELECT product, count FROM sold"
        cursor.execute(query)
        product_results = cursor.fetchall()
        cursor.close()
        connection.close()

        if product_results:
            # Convert data to a DataFrame for better handling
            df = pd.DataFrame(product_results, columns=['Product', 'Count'])

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

            # Create an interactive pie chart using Plotly
            st.subheader("Interactive Product Distribution Pie Chart")
            pie_chart = px.pie(
                df,
                names='Product',
                values='Count',
                title='Product Distribution in Sold Table',
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            pie_chart.update_traces(textinfo='percent+label')
            st.plotly_chart(pie_chart)
        else:
            st.warning("No data available to display charts.")

    elif app_mode=='PRODUCTS':
        # Connect to database
        connection = mysql.connector.connect(host="localhost", user="root", password="", database="shop3")
        cursor = connection.cursor()

        query = "SELECT Name, Category, Stock, Amount FROM products"
        cursor.execute(query)
        res = cursor.fetchall()
        df = pd.DataFrame(res, columns=['Name', 'Category', 'Stock', 'Amount'])

        st.title("Product Details")
        st.write("Available list of products:")

        st.table(df)

        layout, racks = generate_large_mall_layout()

        # Map products to racks
        product_to_shelf = {product: racks[i % len(racks)][0] for i, product in enumerate(df['Name'])}

        st.title("Mall Navigation")
        st.write("Select the products you want to pick:")

        # Display products as checkboxes
        selected_shelves = {}
        for product, shelf in product_to_shelf.items():
            if st.checkbox(f"{product} at shelf {shelf}"):
                selected_shelves[product] = shelf

        if st.button("Start Navigation"):
            if not selected_shelves:
                st.warning("Please select at least one product.")
                return

            G = create_graph_from_layout(layout)

            # Start point
            start_point = (0, 0)

            path = []
            current_location = start_point

            for product, shelf in selected_shelves.items():
                if not is_shelf_accessible(G, shelf):
                    st.error(f"Shelf {shelf} for product {product} is not accessible.")
                    continue

                shortest_path = find_shortest_path_to_product(G, current_location, shelf)
                if shortest_path:
                    path.extend(shortest_path[1:])
                    current_location = shortest_path[-1]
                else:
                    st.error(f"Cannot reach shelf at {shelf} for product {product}.")

            st.write("Shortest Path:", path)

            # Paths to images
            cart_image_path = "cart.jpg"
            rack_image_path = "rack.jpg"
            beacon_image_path = "beacon.png"
            advert_image_path = "advert.png"
            exit_image_path = "exit.jpg"

            visualize_large_layout(
                layout,
                path,
                selected_shelves,
                cart_image_path,
                rack_image_path,
                beacon_image_path,
                advert_image_path,
                exit_image_path
            )

            st.image("mall_path_with_images.png", caption="Shortest Path", use_column_width=True)

    #
    # #---------Product Counter Model---------------------------
    #
    # elif app_mode == 'Shopping Cart' :
    #
    #
    #         st.sidebar.markdown('---')
    #         #use_webcam = st.sidebar.checkbox('Use Webcam')
    #         st.sidebar.info("Upload the video")
    #         st.sidebar.markdown('---')
    #
    #         uploaded_file_top = st.file_uploader("Upload a TOP-view video", type=["mp4", "avi"])
    #         uploaded_file_side = st.file_uploader("Upload a SIDE-view video", type=["mp4", "avi"])
    #
    #         if uploaded_file_top and uploaded_file_side is not None:
    #           # Save the uploaded file to a temporary location
    #             st.sidebar.success("Running Video")
    #             with open("temp_video1.mp4", "wb") as temp_file_top:
    #                 temp_file_top.write(uploaded_file_top.read())
    #
    #             with open("temp_video2.mp4", "wb") as temp_file_side:
    #                 temp_file_side.write(uploaded_file_side.read())
    #
    #
    #         stframe_s = st.empty()
    #         stframe_t = st.empty()
    #
    #         #------------- DISPLAY DETAILS ------------------------------
    #
    #         st.markdown("<hr/>", unsafe_allow_html = True)
    #         kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    #         with kpi1:
    #             st.markdown("**Current ToTal**")
    #             kpi1_text = st.markdown("<h1 style='color:white;'>0</h1>",unsafe_allow_html=True)
    #             #kpi1_text.write(f"<h1 style='text-align:center; color:red;'>0</h1>")
    #         with kpi2:
    #             st.markdown("**Amount**")
    #             #kpi2_text = st.markdown("0")
    #             kpi2_text = st.markdown("<h1 style='color:red;'>0</h1>",unsafe_allow_html=True)
    #         with kpi3:
    #             st.markdown("**Stock**")
    #             #kpi3_text = st.markdown("0")
    #             kpi3_text = st.markdown("<h1 style='color:red;'>0</h1>",unsafe_allow_html=True)
    #         with kpi4:
    #             st.markdown("**Product**")
    #             # kpi3_text = st.markdown("0")
    #             kpi4_text = st.markdown("<h1 style='color:red;'>0</h1>", unsafe_allow_html=True)
    #
    #
    #         st.markdown("<hr/>", unsafe_allow_html = True)
    #         st.markdown("<hr/>", unsafe_allow_html = True)
    #         kpi5,kpi6=st.columns(2)
    #
    #         with kpi5:
    #             #st.markdown("**reciept**")
    #             kpi5_text = st.markdown("<h1 style='color:white;>0</h1>", unsafe_allow_html=True)
    #
    #         st.markdown("<hr/>", unsafe_allow_html=True)
    #
    #         if uploaded_file_top and uploaded_file_side is not None:
    #             load_product_counter(temp_file_side.name,temp_file_top.name, kpi1_text,kpi2_text,  kpi3_text, kpi4_text,kpi5_text,stframe_s,stframe_t)



if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass