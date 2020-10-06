import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

data = pd.read_csv('trump_ratings_first500.csv')
plt.figure(figsize=(15,5))
plt.plot_date(data["datenumber"], data["adjusted_disapprove"])
plt.plot_date(data["datenumber"], data["adjusted_approve"])

poly_parameters_disapprove = np.polyfit(data["datenumber"], data["adjusted_disapprove"], 1)
poly_parameters_approve = np.polyfit(data["datenumber"], data["adjusted_approve"], 1)

my_poly_function_disapprove = np.poly1d(poly_parameters_disapprove)
my_poly_function_approve = np.poly1d(poly_parameters_approve)

expected_y_poly_approve = my_poly_function_approve(data["datenumber"])
expected_y_poly_disapprove = my_poly_function_disapprove(data["datenumber"])                                          

disapprove_fit = np.polyfit(data["datenumber"], data["adjusted_disapprove"], 1)
approve_fit = np.polyfit(data["datenumber"], data["adjusted_approve"], 1)

plt.figure(figsize=(15,5))

plt.plot_date(data["datenumber"], data["adjusted_disapprove"])
plt.plot_date(data["datenumber"], data["adjusted_approve"])

plt.plot(data["datenumber"], expected_y_poly_approve, color = "black", label = "fit")
plt.plot(data["datenumber"], expected_y_poly_disapprove, color = "red", label = "fit")

poly_parameters_disapprove = np.polyfit(data["datenumber"], data["adjusted_disapprove"], 2)
poly_parameters_approve = np.polyfit(data["datenumber"], data["adjusted_approve"], 2)

my_poly_function_disapprove = np.poly1d(poly_parameters_disapprove)
my_poly_function_approve = np.poly1d(poly_parameters_approve)


# Perform higher order fit here to better match the pattern of the data
expected_y_poly_approve = my_poly_function_approve(data["datenumber"])
expected_y_poly_disapprove = my_poly_function_disapprove(data["datenumber"])                                          

plt.figure(figsize=(15,5))

plt.plot_date(data["datenumber"], data["adjusted_disapprove"])
plt.plot_date(data["datenumber"], data["adjusted_approve"])

plt.plot(data["datenumber"], expected_y_poly_approve, color = "black", label = "fit")
plt.plot(data["datenumber"], expected_y_poly_disapprove, color = "red", label = "fit")

x_cos = data["datenumber"]
y_cos_disapprove = data["adjusted_disapprove"]
y_cos_approve = data["adjusted_approve"]


def my_cos_function(x, A, B, C):
    
    return A * np.cos(B * x) + C

popt_disapprove, pcov = curve_fit(my_cos_function, x_cos, y_cos_disapprove)
popt_approve, pcov = curve_fit(my_cos_function, x_cos, y_cos_approve)


a_expected_disapprove = popt_disapprove[0]  # get fitted A value
b_expected_disapprove = popt_disapprove[1]  # get fitted B value
c_expected_disapprove = popt_disapprove[2]  # get fitted C value

a_expected_approve = popt_approve[0]  # get fitted A value
b_expected_approve = popt_approve[1]  # get fitted B value
c_expected_approve = popt_approve[2]  # get fitted C value


y_cos_expected_disapprove = my_cos_function(x_cos, a_expected_disapprove, b_expected_disapprove, c_expected_disapprove)
y_cos_expected_approve = my_cos_function(x_cos, a_expected_approve, b_expected_approve, c_expected_approve)

plt.figure(figsize=(15,5))

plt.scatter(x_cos, y_cos_disapprove, label = "data")


plt.plot(x_cos, y_cos_expected_disapprove, color = "orange", label = "fit")

predict_data = pd.DataFrame({
    'datenumber': np.arange(max(data["datenumber"]), max(data["datenumber"]) + 501, 1)
})
new_data = data.append(predict_data)
x_cos = new_data.datenumber
y_cos_disapprove = data["adjusted_disapprove"]
y_cos_approve = data["adjusted_approve"]


def my_cos_function(x, A, B, C):
    
    return A * np.cos(B * x) + C

popt_disapprove, pcov = curve_fit(my_cos_function, data["datenumber"], y_cos_disapprove)
popt_approve, pcov = curve_fit(my_cos_function, data["datenumber"], y_cos_approve)


a_expected_disapprove = popt_disapprove[0]  # get fitted A value
b_expected_disapprove = popt_disapprove[1]  # get fitted B value
c_expected_disapprove = popt_disapprove[2]  # get fitted C value

a_expected_approve = popt_approve[0]  # get fitted A value
b_expected_approve = popt_approve[1]  # get fitted B value
c_expected_approve = popt_approve[2]  # get fitted C value


y_cos_expected_disapprove = my_cos_function(x_cos, a_expected_disapprove, b_expected_disapprove, c_expected_disapprove)
y_cos_expected_approve = my_cos_function(x_cos, a_expected_approve, b_expected_approve, c_expected_approve)


plt.figure(figsize=(15,5))

y_cos_expected_disapprove
plt.scatter(data["datenumber"], y_cos_disapprove, label = "data")
#plt.scatter(x_cos, y_cos_approve, label = "data")


plt.plot(x_cos, y_cos_expected_disapprove, color = "orange", label = "fit")
