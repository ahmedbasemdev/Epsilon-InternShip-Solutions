import streamlit as st 
from pickle import load
import pandas as pd
import numpy as np
def prediction(model , df):
    predictions_result = model.predict(df)
    return predictions_result

model = load(open('model.pkl', 'rb'))
scaler = load(open('scaler.pkl', 'rb'))
ohe = load(open('OHE_Encoder.pkl', 'rb'))
be = load(open('Binary_Encoder.pkl', 'rb'))
st.title('Banglore Resturants Succes')
st.write('This is a web app to predict the succes of a new restaurant in banglore based on\
        several features that you can see in the sidebar. Please adjust the\
        value of each feature. After that, click on the Predict button at the bottom to\
        see the prediction of the resturant.')
votes         = st.sidebar.slider(label = 'votes', min_value = 0.0,
                        max_value = 16832.0 ,
                        value = 0.0,
                        step = 10.0)
                        
approx_cost_for_two_people = st.sidebar.slider(label = 'approx_cost_for_two_people', min_value = 40.0,
                        max_value = 6000.0 ,
                        value = 400.0,
                        step = 10.0)


cuisines = st.sidebar.slider(label = 'cuisines', min_value = 0.0,
                        max_value = 8.0 ,
                        value = 0.0,
                        step = 1.0)

Online_order = st.selectbox(label = 'Online_order',options = ('Yes', 'No'))
Book_Table = st.selectbox(
     label = 'Book_Table', options = ('Yes', 'No'))
Location = st.selectbox(
     label = 'Location',options = 
     ('Banashankari', 'Basavanagudi', 'other', 'Jayanagar', 'JP Nagar',
       'Bannerghatta Road', 'BTM', 'Electronic City', 'Shanti Nagar',
       'Koramangala 5th Block', 'Richmond Road', 'HSR',
       'Koramangala 7th Block', 'Bellandur', 'Sarjapur Road',
       'Marathahalli', 'Whitefield', 'Old Airport Road', 'Indiranagar',
       'Koramangala 1st Block', 'Frazer Town', 'MG Road', 'Brigade Road',
       'Lavelle Road', 'Church Street', 'Ulsoor', 'Residency Road',
       'Shivajinagar', 'St. Marks Road', 'Cunningham Road',
       'Commercial Street', 'Vasanth Nagar', 'Domlur',
       'Koramangala 8th Block', 'Ejipura', 'Jeevan Bhima Nagar',
       'Kammanahalli', 'Koramangala 6th Block', 'Brookefield',
       'Koramangala 4th Block', 'Banaswadi', 'Kalyan Nagar',
       'Malleshwaram', 'Rajajinagar', 'New BEL Road'))
rest_type = st.selectbox(
     label = 'rest_type',options = 
     ('Casual Dining', 'other', 'Quick Bites', 'Cafe', 'Delivery',
       'Dessert Parlor', 'Bakery', 'Takeaway, Delivery', 'Beverage Shop',
       'Bar', 'Casual Dining, Bar', 'Food Court'))
listed_in_type = st.selectbox(
     label = 'listed_in(type)',options = 
     ('Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out',
       'Drinks & nightlife', 'Pubs and bars'))
listed_in_city = st.selectbox(
     label = 'listed_in(city)',options = 
     ('Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
       'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
       'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
       'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
       'Koramangala 4th Block', 'Koramangala 5th Block',
       'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
       'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
       'Old Airport Road', 'Rajajinagar', 'Residency Road',
       'Sarjapur Road', 'Whitefield'))
	 
features = {
    'votes':votes ,
    'approx_cost(for two people)':approx_cost_for_two_people,
    "online_order":Online_order,
    "book_table" : Book_Table,
    "location":Location,
    "rest_type":rest_type,
    "listed_in(type)":listed_in_type,
    "listed_in(city)":listed_in_city,
    'cuisines':cuisines
    
}
features = pd.DataFrame([features])
def count_types(r):

    return len(r.split(','))

features['total_types'] = features['rest_type'].apply(count_types)

xy = ohe.transform(features[['online_order','book_table','listed_in(type)']])

xy_df = pd.DataFrame(xy , columns=ohe.get_feature_names_out())

features  = pd.concat([features,xy_df] , axis = 1)

features.drop(['online_order','book_table','listed_in(type)'] , axis =1 , inplace = True)

be_df= be.transform(features[['location','rest_type','listed_in(city)' ]])

features = pd.concat([features,be_df] , axis = 1)

features.drop(['location','rest_type','listed_in(city)' ], axis = 1 , inplace = True)

features[['votes','approx_cost(for two people)']] = scaler.transform(features[['votes','approx_cost(for two people)']])



if st.button('Predict'):
    
    prediction = prediction(model, features)
    st.write(' Based on feature values, the resturant is '+ str(prediction))