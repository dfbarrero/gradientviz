import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go

import numpy as np
import pandas as pd

side_len = 10
fig_size = (side_len, side_len)
quiver_alpha=.75


# n_points = 180
# exploring resolutions from this point on
# n_points = 73
# n_points = 110
#n_points = 115
n_points = 220 # best yet!

lim = 2
y_bar = 0.2
spacing = (lim-(-lim))/n_points
x = np.linspace(-lim, lim, n_points)
y = np.linspace(-lim, lim, n_points)
xv, yv = np.meshgrid(x, y)

@np.vectorize
def mae(x, y):
    return np.abs(x-y)

def mae2(x, y):
    return np.abs(x-y)

@np.vectorize
def mbe(x, y):
    return x-y

@np.vectorize
def mse(x, y):
    return (x-y)**2

@np.vectorize
def rae(x, y):
    return np.abs(x-y)/(np.abs(x-y_bar)+np.finfo(np.float32).eps)
#     return np.abs(x-y)/np.abs(x-y_bar)

@np.vectorize
def rse(x, y):
    return (x-y)**2/((x-y_bar)**2+np.finfo(np.float32).eps)
#     return (x-y)**2/(x-y_bar)**2

@np.vectorize
def mape(x, y):
    return np.abs(x-y)/(np.abs(x)+np.finfo(np.float32).eps)
#     return np.abs(x-y)/np.abs(x)

@np.vectorize
def smape(x, y):
    return np.abs(x-y)/((np.abs(x)+np.abs(y))/2+np.finfo(np.float32).eps)
#     return np.abs(x-y)/((np.abs(x)+np.abs(y))/2)

@np.vectorize
def logcosh(x, y):
    return np.log10(np.cosh(y-x))

@np.vectorize
def huber(x, y, delta):
    diff = np.abs(x-y)
    if diff > delta:
        return 1/2*diff**2
    else:
        return delta*diff-delta/2
    
@np.vectorize
def quantile(x, y, gamma):
    if x < y:
        return (1-gamma)*mae2(x,y)
    else:
        return gamma*mae2(x,y)

metrics = {"mae": mae, 
        "mse": mse, 
        "rse": rse, 
        "mape": mape, 
        "smape": smape,
        'logcosh': logcosh,
        "rae": rae,
        "mbe": mbe,
        "huber": huber,
        "quantile": quantile
        }

labels = {"mae": "Mean absolute error (MAE)",
          "mse": "Mean squared error (MSE)",
          "rse": "Root squared error (RSE)",
          "mape": "Mean absolute percentage error (MAPE)",
          "smape": "Simmetric mean absolute percentage error (SMAPE)",
          "logcosh": "Log-cosh error",
          "rae": "RAE",
          "mbe": "Mean Bias Error (MBE)",
          "huber": "Huber",
          "quantile": "Quantile"
          }

definitions = {"mae": "$MAE(Y, X; \hat{f}_\\beta) = \\frac{1}{N}\sum_{i=1}^N|y_i - \hat{f}_\\beta(x_i)|$",
          "mse": "$\\textnormal{MSE}(Y, X; \hat{f}_\\beta) = \\frac{\sum_{i=1}^N (y_i - \hat{f}_\\beta (X_i))^2}{N}$",
          "rse": "TODO Root squared error (RSE)",
          "mape": "TODO Mean absolute percentage error (MAPE)",
          "smape": "TODO immetric mean absolute percentage error (SMAPE)",
          "logcosh": "TODO Log-cosh error",
          "rae": "TODO RAE",
          "mbe": "TODO Mean Bias Error (MBE)",
          "huber": "TODO Huber",
          "quantile": "TODO Quantile"
          }

gradients = {"mae": "$\\nabla MAE(Y, X; \hat{f}_\\beta) \simeq \\frac{1}{N}\sum_{i=1}^N\\textnormal{sgn}(l_i)$",
          "mse": "Let $l_i$ denote model $\hat{f}_\\beta$'s error for an input-target pair $(X_i, Y_i)$, then $\\nabla MSE \simeq \\frac{1}{N}\sum_{i=1}^N2l_i$",
          "rse": "TODO Root squared error (RSE)",
          "mape":" TODO Mean absolute percentage error (MAPE)",
          "smape": "STODO immetric mean absolute percentage error (SMAPE)",
          "logcosh": "TODO Log-cosh error",
          "rae": "TODO RAE",
          "mbe": "TODO Mean Bias Error (MBE)",
          "huber": "TODO Huber",
          "quantile": "TODO Quantile"
        }

checkboxes = {}

colorscale="magenta"

st.write("# Error gradient visualization")
selectedMetrics = st.sidebar.multiselect(
    "Metrics to show",
    ["MAE", "MSE", "RSE", "MAPE", "SMAPE", "logcosh", "RAE", "MBE", "Huber", "Quantile"],
    ["MAE", "MSE", "RSE", "MAPE", "SMAPE", "logcosh", "RAE", "MBE", "Huber", "Quantile"])

#st.sidebar.write("Metrics")
#for k in metrics.keys():
#    checkboxes[k] = st.sidebar.checkbox(k.upper(), value=True)

st.sidebar.write("Plots")
surface = st.sidebar.toggle("Surface", value=True)
contour = st.sidebar.toggle("Contour", value=False)
#mesh = st.sidebar.toggle("Mesh", value=False)

st.sidebar.write("Equations")
showdefinition = st.sidebar.toggle("Definitions", value=True)
showgradients = st.sidebar.toggle("Gradients", value=True)


if showdefinition or showgradients:
    st.write("Let $\hat{f}_\\beta(x)$ denote a regressor parameterized by $\\beta$")


for metric in [x.lower() for x in selectedMetrics]:
    st.subheader(labels[metric])

    if showdefinition: st.write(definitions[metric])
    if showgradients: st.write(gradients[metric])

    match metric:
        case "huber":
            delta =  st.slider('$\delta$', min_value=0.5, max_value=1.5, value=1.0, step=0.5) 
            z = metrics[metric](xv.T, yv.T, delta)
        case "quantile":
            gamma =  st.slider('$\gamma$', min_value=0.0, max_value=1.0, value=0.5, step=0.25)
            z = metrics[metric](xv.T, yv.T, gamma)
        case _: 
            z = metrics[metric](xv.T, yv.T)

    if surface:
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale=colorscale)])
        fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                 highlightcolor="limegreen", project_z=True))
                 #project_z=True))

        fig.update_layout(autosize=False,
                 width=500, height=500,
                 margin=dict(l=65, r=50, b=65, t=90))
        st.plotly_chart(fig, use_container_witth=True)

    if contour:
        cont = go.Figure(data=[go.Contour(x=y, y=y, z=z, colorscale=colorscale)])
        st.plotly_chart(cont, use_container_witth=True)

    mesh = False
    if mesh:
        mesh = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z)])
        st.plotly_chart(mesh, use_container_witth=True)
