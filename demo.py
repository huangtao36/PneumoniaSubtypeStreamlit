import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from PIL import Image
import numpy as np
# from multiapp import MultiApp
import joblib
# from sklearn.externals import joblib
import streamlit as st
import matplotlib.pyplot as plt

sns.set(style='ticks')
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', family='FangSong', weight='bold')  # 用于画图时显示中文

warnings.filterwarnings('ignore')


@st.cache(suppress_st_warning=True)
def dis_img1():
    image1 = Image.open('./pneu_img/聚类散点密度图2.png')  # Image name
    fig = plt.figure()
    plt.imshow(image1)
    plt.axis("off")

    image2 = Image.open('./pneu_img/Subphenotype_Status1.png')  # Image name
    fig = plt.figure()
    plt.imshow(image2)
    plt.axis("off")
    return image1, image2


image1, image2 = dis_img1()


@st.cache(suppress_st_warning=True)
def dis_img2():
    image1 = Image.open('./pneu_img/流程图.png')  # Image name
    fig = plt.figure()
    plt.imshow(image1)
    plt.axis("off")
    return image1


image3 = dis_img2()


def show_dashbord():
    st.markdown(
        ":point_left: **Open the right bar and enter health indicators using the pneumonia subtype identification tool** :grey_exclamation:")

    st.title('Subtype Identification and Analysis of Patients with Severe Pneumonia')
    st.markdown(
        'We developed a subtype recognition model for severe pneumonia based on [MIMIC-IV](https://mimic.mit.edu/) to distinguish patients and further guide physicians to provide appropriate treatment and improve patient survival.')

    col1, col2 = st.columns([2, 3])
    with col1:
        st.image(image1, width=250, caption="Clustering Distribution")
        st.markdown(
            'The results showed that patients with severe pneumonia could be divided into two clinical subtypes.')
        st.markdown('\n \n')

        st.image(image2, width=250, caption="Number of Subtypes and Survival Probability")
        st.markdown('There were significant differences in survival rate and survival time between the two subtypes.')

    with col2:
        st.image(image3, caption="Flow Chart")
        st.markdown('Retrospective analysis showed that there were also significant differences '
                    'in medication between the two subtypes, which further proved the availability of '
                    'this subtype recognition model. This is described in more detail in our paper.')

    st.markdown(
        ":point_left: **Open the right bar and enter health indicators using the pneumonia subtype identification tool** :grey_exclamation:")


st.sidebar.title("Enter your health indicators: ")

age = st.sidebar.number_input('Age (years): ', min_value=18, max_value=89, value=18, step=1)
bmi = st.sidebar.number_input('BMI: ', min_value=11.72, max_value=47.87, value=32.33, step=0.01)
apsiii = st.sidebar.slider('APSIII (scores): ', min_value=0, max_value=184, value=77, step=1)
sofa = st.sidebar.slider('SOFA (scores): ', min_value=0, max_value=19, value=13, step=1)
options = st.sidebar.multiselect('Have any of the following diseases: ',
                                 ['Cancer', 'Cerebrovascular', 'Hypertension'])
st.sidebar.markdown("Is there any situation as follows: ")
vent = st.sidebar.checkbox('Invasive Ventilator', value=False)
bs = st.sidebar.checkbox('Blood Sputum Microbiological Test', value=False)
nibp = st.sidebar.number_input('Non Invasive Blood Pressure systolic (mmHg): ', min_value=53, max_value=189, value=144,
                               step=1)
po2 = st.sidebar.number_input('pO2 (mmHg): ', min_value=8, max_value=361, value=322, step=1)
o2 = st.sidebar.number_input('O2 saturation pulseoxymetry (%): ', min_value=88, max_value=100, value=88, step=1)
o2f = st.sidebar.number_input('O2 Flow (L/min): ', min_value=0.5, max_value=20.0, value=6.0, step=0.5)
co2 = st.sidebar.number_input('Arterial CO2 Pressure (mmHg): ', min_value=14, max_value=70, value=34, step=1)
pco2 = st.sidebar.number_input('pCO2 (mmHg): ', min_value=13.5, max_value=33.0, value=23.5, step=0.5)
chloride = st.sidebar.number_input('Chloride (mEq/L): ', min_value=86, max_value=121, value=98, step=1)
sodium = st.sidebar.number_input('Sodium (mEq/L): ', min_value=129, max_value=148, value=134, step=1)
bicarbonate = st.sidebar.number_input('Bicarbonate (mEq/L): ', min_value=13, max_value=33, value=26, step=1)
plr = st.sidebar.number_input('Pain Level Response (%): ', min_value=1, max_value=7, value=5, step=1)
albumin = st.sidebar.number_input('Albumin (g/dL): ', min_value=1.2, max_value=4.7, value=2.6, step=0.1)
wbc = st.sidebar.number_input('White Blood Cells (K/uL): ', min_value=0.1, max_value=26.7, value=19.8, step=0.1)
glucose = st.sidebar.number_input('Glucose (mg/dL): ', min_value=14, max_value=254, value=189, step=1)
rr = st.sidebar.number_input('Respiratory Rate (insp/min): ', min_value=4, max_value=36, value=23, step=1)
temp = st.sidebar.number_input('Temperature Fahrenheit (°F): ', min_value=96.1, max_value=100.4, value=97.3, step=0.1)

predict = st.sidebar.button("Predict")
back = st.sidebar.button("Back Home")

if predict:
    # ['BMI', '220179', '220235', 'hypertension', '223761', 'cancer', '51301',
    #   '50902', '50931', 'age', '220210', '50882', '224409', '50983', '50821',
    #   'BLOODSPUTUM', '50818', 'apsiii', '50862', 'cerebrovascular_disease',
    #   '223834', 'sofa', '220277', 'InvasiveVent']

    Cancer, Cerebrovascular, Hypertension = 0, 0, 0
    if 'Cancer' in options:
        Cancer = 1
    if 'Cerebrovascular' in options:
        Cerebrovascular = 1
    if 'Hypertension' in options:
        Hypertension = 1
    InvasiveVentilator, Microbiological = 0, 0
    if vent: InvasiveVentilator = 1
    if bs: Microbiological = 1

    data_list = [bmi, nibp, co2, Hypertension, temp, Cancer, wbc,
                 chloride, glucose, age, rr, bicarbonate, plr, sodium, po2,
                 Microbiological, pco2, apsiii, albumin, Cerebrovascular,
                 o2f, sofa, o2, InvasiveVentilator]
    data_df = pd.DataFrame(np.array(data_list)).T

    mean_std_df = pd.read_csv('./mean_std_df.csv')
    data_df = pd.DataFrame((np.array(data_list) - mean_std_df['mean']) / mean_std_df['std']).T

    data_df.columns = ['BMI', '220179', '220235', 'hypertension', '223761', 'cancer', '51301',
                       '50902', '50931', 'age', '220210', '50882', '224409', '50983', '50821',
                       'BLOODSPUTUM', '50818', 'apsiii', '50862', 'cerebrovascular_disease',
                       '223834', 'sofa', '220277', 'InvasiveVent']
    st.markdown("## :bell: We Got the Result: :clap:")
    # st.dataframe(data_df)
    # st.write(data_df.shape)

    lr_model = joblib.load('lr.model')
    proba = lr_model.predict_proba(data_df)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(":point_right: The probability of patients belonging to subtype :one: was: ")
        st.markdown("## **%.3f**" % proba[0, 0])

        st.write(":point_right: The probability of patients belonging to subtype :two: was: ")
        st.markdown("## **%.3f**" % proba[0, 1])
    with col2:
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'Subtype 1', 'Subtype 2'
        sizes = [proba[0, 0], proba[0, 1]]
        explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                                           shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # 更改外围的文字大小、颜色
        for text in texts:
            text.set_size(20)
            text.set_color('navy')
        # 更改饼图内的文字大小、颜色
        for text in autotexts:
            text.set_size(20)
        st.pyplot(fig1, dpi=800)

    # col1, col2 = st.columns([2, 3])
    # with col1:
    #     st.markdown("### 亚型1的主要特征")
    #
    # with col2:
    #     st.markdown("### 亚型2的主要特征")


else:
    show_dashbord()

if back and predict:
    show_dashbord()
