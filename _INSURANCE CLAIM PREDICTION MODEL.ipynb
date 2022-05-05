{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## CAPSTONE PROJECT 3\n",
    "### TEAM ID: \n",
    "#### PTID-CDS-JUL21-1172\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n",
    "### PROJECT ID:\n",
    "#### PRCP-1010-InsClaimPred\n",
    "### PROJECT NAME:\n",
    "#### Insurance Claim Prediction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "OBJECTIVE: The target here is to build a Machine Learning model that predicts the probability that a driver will initiate an insurance claim in the following year.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Introduction:\n",
    "This is an Insurance Claim Prediction Data Set. The bigger scope is to account for inaccuracies in the cost of insurance policies. A cautious driver should get a better price compared to a reckless driver. The aim is to build a machine learning model to predict whether an auto insurance policy holder files a claim.\n",
    "### Dataset description:\n",
    "In data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## IMPORTING NECESSARY LIBRARIES"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## LOADING THE DATASET"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'C:\\\\Users\\\\arshad'"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pwd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C:\\Users\\arshad\\Downloads\\PRCP-1010-InsClaimPred\\Data\n"
     ]
    }
   ],
   "source": [
    "cd \"C:\\Users\\arshad\\Downloads\\PRCP-1010-InsClaimPred\\Data\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "data=pd.read_csv('train.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## EXPLORATORY DATA ANALYSIS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>target</th>\n",
       "      <th>ps_ind_01</th>\n",
       "      <th>ps_ind_02_cat</th>\n",
       "      <th>ps_ind_03</th>\n",
       "      <th>ps_ind_04_cat</th>\n",
       "      <th>ps_ind_05_cat</th>\n",
       "      <th>ps_ind_06_bin</th>\n",
       "      <th>ps_ind_07_bin</th>\n",
       "      <th>ps_ind_08_bin</th>\n",
       "      <th>...</th>\n",
       "      <th>ps_calc_11</th>\n",
       "      <th>ps_calc_12</th>\n",
       "      <th>ps_calc_13</th>\n",
       "      <th>ps_calc_14</th>\n",
       "      <th>ps_calc_15_bin</th>\n",
       "      <th>ps_calc_16_bin</th>\n",
       "      <th>ps_calc_17_bin</th>\n",
       "      <th>ps_calc_18_bin</th>\n",
       "      <th>ps_calc_19_bin</th>\n",
       "      <th>ps_calc_20_bin</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>13</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>7</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>16</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>17</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 59 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   id  target  ps_ind_01  ps_ind_02_cat  ps_ind_03  ps_ind_04_cat  \\\n",
       "0   7       0          2              2          5              1   \n",
       "1   9       0          1              1          7              0   \n",
       "2  13       0          5              4          9              1   \n",
       "3  16       0          0              1          2              0   \n",
       "4  17       0          0              2          0              1   \n",
       "\n",
       "   ps_ind_05_cat  ps_ind_06_bin  ps_ind_07_bin  ps_ind_08_bin  ...  \\\n",
       "0              0              0              1              0  ...   \n",
       "1              0              0              0              1  ...   \n",
       "2              0              0              0              1  ...   \n",
       "3              0              1              0              0  ...   \n",
       "4              0              1              0              0  ...   \n",
       "\n",
       "   ps_calc_11  ps_calc_12  ps_calc_13  ps_calc_14  ps_calc_15_bin  \\\n",
       "0           9           1           5           8               0   \n",
       "1           3           1           1           9               0   \n",
       "2           4           2           7           7               0   \n",
       "3           2           2           4           9               0   \n",
       "4           3           1           1           3               0   \n",
       "\n",
       "   ps_calc_16_bin  ps_calc_17_bin  ps_calc_18_bin  ps_calc_19_bin  \\\n",
       "0               1               1               0               0   \n",
       "1               1               1               0               1   \n",
       "2               1               1               0               1   \n",
       "3               0               0               0               0   \n",
       "4               0               0               1               1   \n",
       "\n",
       "   ps_calc_20_bin  \n",
       "0               1  \n",
       "1               0  \n",
       "2               0  \n",
       "3               0  \n",
       "4               0  \n",
       "\n",
       "[5 rows x 59 columns]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 595212 entries, 0 to 595211\n",
      "Data columns (total 59 columns):\n",
      " #   Column          Non-Null Count   Dtype  \n",
      "---  ------          --------------   -----  \n",
      " 0   id              595212 non-null  int64  \n",
      " 1   target          595212 non-null  int64  \n",
      " 2   ps_ind_01       595212 non-null  int64  \n",
      " 3   ps_ind_02_cat   595212 non-null  int64  \n",
      " 4   ps_ind_03       595212 non-null  int64  \n",
      " 5   ps_ind_04_cat   595212 non-null  int64  \n",
      " 6   ps_ind_05_cat   595212 non-null  int64  \n",
      " 7   ps_ind_06_bin   595212 non-null  int64  \n",
      " 8   ps_ind_07_bin   595212 non-null  int64  \n",
      " 9   ps_ind_08_bin   595212 non-null  int64  \n",
      " 10  ps_ind_09_bin   595212 non-null  int64  \n",
      " 11  ps_ind_10_bin   595212 non-null  int64  \n",
      " 12  ps_ind_11_bin   595212 non-null  int64  \n",
      " 13  ps_ind_12_bin   595212 non-null  int64  \n",
      " 14  ps_ind_13_bin   595212 non-null  int64  \n",
      " 15  ps_ind_14       595212 non-null  int64  \n",
      " 16  ps_ind_15       595212 non-null  int64  \n",
      " 17  ps_ind_16_bin   595212 non-null  int64  \n",
      " 18  ps_ind_17_bin   595212 non-null  int64  \n",
      " 19  ps_ind_18_bin   595212 non-null  int64  \n",
      " 20  ps_reg_01       595212 non-null  float64\n",
      " 21  ps_reg_02       595212 non-null  float64\n",
      " 22  ps_reg_03       595212 non-null  float64\n",
      " 23  ps_car_01_cat   595212 non-null  int64  \n",
      " 24  ps_car_02_cat   595212 non-null  int64  \n",
      " 25  ps_car_03_cat   595212 non-null  int64  \n",
      " 26  ps_car_04_cat   595212 non-null  int64  \n",
      " 27  ps_car_05_cat   595212 non-null  int64  \n",
      " 28  ps_car_06_cat   595212 non-null  int64  \n",
      " 29  ps_car_07_cat   595212 non-null  int64  \n",
      " 30  ps_car_08_cat   595212 non-null  int64  \n",
      " 31  ps_car_09_cat   595212 non-null  int64  \n",
      " 32  ps_car_10_cat   595212 non-null  int64  \n",
      " 33  ps_car_11_cat   595212 non-null  int64  \n",
      " 34  ps_car_11       595212 non-null  int64  \n",
      " 35  ps_car_12       595212 non-null  float64\n",
      " 36  ps_car_13       595212 non-null  float64\n",
      " 37  ps_car_14       595212 non-null  float64\n",
      " 38  ps_car_15       595212 non-null  float64\n",
      " 39  ps_calc_01      595212 non-null  float64\n",
      " 40  ps_calc_02      595212 non-null  float64\n",
      " 41  ps_calc_03      595212 non-null  float64\n",
      " 42  ps_calc_04      595212 non-null  int64  \n",
      " 43  ps_calc_05      595212 non-null  int64  \n",
      " 44  ps_calc_06      595212 non-null  int64  \n",
      " 45  ps_calc_07      595212 non-null  int64  \n",
      " 46  ps_calc_08      595212 non-null  int64  \n",
      " 47  ps_calc_09      595212 non-null  int64  \n",
      " 48  ps_calc_10      595212 non-null  int64  \n",
      " 49  ps_calc_11      595212 non-null  int64  \n",
      " 50  ps_calc_12      595212 non-null  int64  \n",
      " 51  ps_calc_13      595212 non-null  int64  \n",
      " 52  ps_calc_14      595212 non-null  int64  \n",
      " 53  ps_calc_15_bin  595212 non-null  int64  \n",
      " 54  ps_calc_16_bin  595212 non-null  int64  \n",
      " 55  ps_calc_17_bin  595212 non-null  int64  \n",
      " 56  ps_calc_18_bin  595212 non-null  int64  \n",
      " 57  ps_calc_19_bin  595212 non-null  int64  \n",
      " 58  ps_calc_20_bin  595212 non-null  int64  \n",
      "dtypes: float64(10), int64(49)\n",
      "memory usage: 267.9 MB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>target</th>\n",
       "      <th>ps_ind_01</th>\n",
       "      <th>ps_ind_02_cat</th>\n",
       "      <th>ps_ind_03</th>\n",
       "      <th>ps_ind_04_cat</th>\n",
       "      <th>ps_ind_05_cat</th>\n",
       "      <th>ps_ind_06_bin</th>\n",
       "      <th>ps_ind_07_bin</th>\n",
       "      <th>ps_ind_08_bin</th>\n",
       "      <th>...</th>\n",
       "      <th>ps_calc_11</th>\n",
       "      <th>ps_calc_12</th>\n",
       "      <th>ps_calc_13</th>\n",
       "      <th>ps_calc_14</th>\n",
       "      <th>ps_calc_15_bin</th>\n",
       "      <th>ps_calc_16_bin</th>\n",
       "      <th>ps_calc_17_bin</th>\n",
       "      <th>ps_calc_18_bin</th>\n",
       "      <th>ps_calc_19_bin</th>\n",
       "      <th>ps_calc_20_bin</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>5.952120e+05</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "      <td>595212.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>7.438036e+05</td>\n",
       "      <td>0.036448</td>\n",
       "      <td>1.900378</td>\n",
       "      <td>1.358943</td>\n",
       "      <td>4.423318</td>\n",
       "      <td>0.416794</td>\n",
       "      <td>0.405188</td>\n",
       "      <td>0.393742</td>\n",
       "      <td>0.257033</td>\n",
       "      <td>0.163921</td>\n",
       "      <td>...</td>\n",
       "      <td>5.441382</td>\n",
       "      <td>1.441918</td>\n",
       "      <td>2.872288</td>\n",
       "      <td>7.539026</td>\n",
       "      <td>0.122427</td>\n",
       "      <td>0.627840</td>\n",
       "      <td>0.554182</td>\n",
       "      <td>0.287182</td>\n",
       "      <td>0.349024</td>\n",
       "      <td>0.153318</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>4.293678e+05</td>\n",
       "      <td>0.187401</td>\n",
       "      <td>1.983789</td>\n",
       "      <td>0.664594</td>\n",
       "      <td>2.699902</td>\n",
       "      <td>0.493311</td>\n",
       "      <td>1.350642</td>\n",
       "      <td>0.488579</td>\n",
       "      <td>0.436998</td>\n",
       "      <td>0.370205</td>\n",
       "      <td>...</td>\n",
       "      <td>2.332871</td>\n",
       "      <td>1.202963</td>\n",
       "      <td>1.694887</td>\n",
       "      <td>2.746652</td>\n",
       "      <td>0.327779</td>\n",
       "      <td>0.483381</td>\n",
       "      <td>0.497056</td>\n",
       "      <td>0.452447</td>\n",
       "      <td>0.476662</td>\n",
       "      <td>0.360295</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>7.000000e+00</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>3.719915e+05</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>7.435475e+05</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.115549e+06</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.488027e+06</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>11.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>13.000000</td>\n",
       "      <td>23.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>8 rows × 59 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                 id         target      ps_ind_01  ps_ind_02_cat  \\\n",
       "count  5.952120e+05  595212.000000  595212.000000  595212.000000   \n",
       "mean   7.438036e+05       0.036448       1.900378       1.358943   \n",
       "std    4.293678e+05       0.187401       1.983789       0.664594   \n",
       "min    7.000000e+00       0.000000       0.000000      -1.000000   \n",
       "25%    3.719915e+05       0.000000       0.000000       1.000000   \n",
       "50%    7.435475e+05       0.000000       1.000000       1.000000   \n",
       "75%    1.115549e+06       0.000000       3.000000       2.000000   \n",
       "max    1.488027e+06       1.000000       7.000000       4.000000   \n",
       "\n",
       "           ps_ind_03  ps_ind_04_cat  ps_ind_05_cat  ps_ind_06_bin  \\\n",
       "count  595212.000000  595212.000000  595212.000000  595212.000000   \n",
       "mean        4.423318       0.416794       0.405188       0.393742   \n",
       "std         2.699902       0.493311       1.350642       0.488579   \n",
       "min         0.000000      -1.000000      -1.000000       0.000000   \n",
       "25%         2.000000       0.000000       0.000000       0.000000   \n",
       "50%         4.000000       0.000000       0.000000       0.000000   \n",
       "75%         6.000000       1.000000       0.000000       1.000000   \n",
       "max        11.000000       1.000000       6.000000       1.000000   \n",
       "\n",
       "       ps_ind_07_bin  ps_ind_08_bin  ...     ps_calc_11     ps_calc_12  \\\n",
       "count  595212.000000  595212.000000  ...  595212.000000  595212.000000   \n",
       "mean        0.257033       0.163921  ...       5.441382       1.441918   \n",
       "std         0.436998       0.370205  ...       2.332871       1.202963   \n",
       "min         0.000000       0.000000  ...       0.000000       0.000000   \n",
       "25%         0.000000       0.000000  ...       4.000000       1.000000   \n",
       "50%         0.000000       0.000000  ...       5.000000       1.000000   \n",
       "75%         1.000000       0.000000  ...       7.000000       2.000000   \n",
       "max         1.000000       1.000000  ...      19.000000      10.000000   \n",
       "\n",
       "          ps_calc_13     ps_calc_14  ps_calc_15_bin  ps_calc_16_bin  \\\n",
       "count  595212.000000  595212.000000   595212.000000   595212.000000   \n",
       "mean        2.872288       7.539026        0.122427        0.627840   \n",
       "std         1.694887       2.746652        0.327779        0.483381   \n",
       "min         0.000000       0.000000        0.000000        0.000000   \n",
       "25%         2.000000       6.000000        0.000000        0.000000   \n",
       "50%         3.000000       7.000000        0.000000        1.000000   \n",
       "75%         4.000000       9.000000        0.000000        1.000000   \n",
       "max        13.000000      23.000000        1.000000        1.000000   \n",
       "\n",
       "       ps_calc_17_bin  ps_calc_18_bin  ps_calc_19_bin  ps_calc_20_bin  \n",
       "count   595212.000000   595212.000000   595212.000000   595212.000000  \n",
       "mean         0.554182        0.287182        0.349024        0.153318  \n",
       "std          0.497056        0.452447        0.476662        0.360295  \n",
       "min          0.000000        0.000000        0.000000        0.000000  \n",
       "25%          0.000000        0.000000        0.000000        0.000000  \n",
       "50%          1.000000        0.000000        0.000000        0.000000  \n",
       "75%          1.000000        1.000000        1.000000        0.000000  \n",
       "max          1.000000        1.000000        1.000000        1.000000  \n",
       "\n",
       "[8 rows x 59 columns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='target', ylabel='count'>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZgAAAEKCAYAAAAvlUMdAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAV3klEQVR4nO3df6zd9X3f8ecrdkrIEogNF0ZtqFnxsgFryLgytJmqNp5sT11rlELqqhlWZs0bZVkqTd1gquYJxhS0bFmIApI1HAzrCp7XDjcKZZ5pFmUjgJ2m41eQvUDBgmLCdYBsg8b0vT/O58bHl+PLtePPvcZ+PqSj8z3v7/fzOZ+vZHjp8/18z/emqpAk6Vh711wPQJJ0YjJgJEldGDCSpC4MGElSFwaMJKkLA0aS1EXXgEnygSRbk3w7yZNJfjrJwiTbk+xu7wuGjr8+yZ4kTyVZOVS/NMmjbd8tSdLqpyS5p9UfSrJkqM3a9h27k6zteZ6SpLfqPYP5PPAHVfVXgA8BTwLXATuqaimwo30myYXAGuAiYBVwa5J5rZ/bgPXA0vZa1errgP1VdQHwOeDm1tdCYANwGbAM2DAcZJKk/roFTJLTgJ8Fbgeoqj+rqu8Bq4HN7bDNwBVtezVwd1W9UVVPA3uAZUnOAU6rqgdr8KvQO6e0mexrK7C8zW5WAturaqKq9gPbORhKkqRZML9j338JeAn4UpIPAbuATwNnV9ULAFX1QpKz2vGLgG8Mtd/baj9o21Prk22ea30dSPIKcMZwfUSbkc4888xasmTJEZ6iJJ3cdu3a9d2qGhu1r2fAzAf+OvCpqnooyedpl8MOIyNqNU39aNsc/MJkPYNLb5x33nns3LlzmuFJkqZK8ieH29dzDWYvsLeqHmqftzIInBfbZS/a+76h488dar8YeL7VF4+oH9ImyXzgdGBimr4OUVUbq2q8qsbHxkYGsCTpKHULmKr6U+C5JB9speXAE8A2YPKurrXAvW17G7Cm3Rl2PoPF/Ifb5bTXklze1leuntJmsq8rgQfaOs39wIokC9ri/opWkyTNkp6XyAA+Bfx2kh8DvgN8kkGobUmyDngWuAqgqh5PsoVBCB0Arq2qN1s/1wB3AKcC97UXDG4guCvJHgYzlzWtr4kkNwKPtONuqKqJnicqSTpUfFz/wPj4eLkGI0lHJsmuqhoftc9f8kuSujBgJEldGDCSpC4MGElSFwaMJKmL3rcpn1Qu/c0753oIOg7t+tdXz/UQpDnhDEaS1IUBI0nqwoCRJHVhwEiSujBgJEldGDCSpC4MGElSFwaMJKkLA0aS1IUBI0nqwoCRJHVhwEiSujBgJEldGDCSpC4MGElSFwaMJKkLA0aS1IUBI0nqwoCRJHVhwEiSujBgJElddA2YJM8keTTJt5LsbLWFSbYn2d3eFwwdf32SPUmeSrJyqH5p62dPkluSpNVPSXJPqz+UZMlQm7XtO3YnWdvzPCVJbzUbM5ifr6pLqmq8fb4O2FFVS4Ed7TNJLgTWABcBq4Bbk8xrbW4D1gNL22tVq68D9lfVBcDngJtbXwuBDcBlwDJgw3CQSZL6m4tLZKuBzW17M3DFUP3uqnqjqp4G9gDLkpwDnFZVD1ZVAXdOaTPZ11ZgeZvdrAS2V9VEVe0HtnMwlCRJs6B3wBTwX5PsSrK+1c6uqhcA2vtZrb4IeG6o7d5WW9S2p9YPaVNVB4BXgDOm6UuSNEvmd+7/I1X1fJKzgO1Jvj3NsRlRq2nqR9vm4BcOQm89wHnnnTfN0CRJR6rrDKaqnm/v+4DfY7Ae8mK77EV739cO3wucO9R8MfB8qy8eUT+kTZL5wOnAxDR9TR3fxqoar6rxsbGxoz9RSdJbdAuYJH8hyfsnt4EVwGPANmDyrq61wL1texuwpt0Zdj6DxfyH22W015Jc3tZXrp7SZrKvK4EH2jrN/cCKJAva4v6KVpMkzZKel8jOBn6v3VE8H/iPVfUHSR4BtiRZBzwLXAVQVY8n2QI8ARwArq2qN1tf1wB3AKcC97UXwO3AXUn2MJi5rGl9TSS5EXikHXdDVU10PFdJ0hTdAqaqvgN8aET9ZWD5YdrcBNw0or4TuHhE/XVaQI3YtwnYdGSjliQdK/6SX5LUhQEjSerCgJEkdWHASJK6MGAkSV0YMJKkLgwYSVIXBowkqQsDRpLUhQEjSerCgJEkdWHASJK6MGAkSV0YMJKkLgwYSVIXBowkqQsDRpLUhQEjSerCgJEkdWHASJK6MGAkSV0YMJKkLgwYSVIXBowkqQsDRpLUhQEjSerCgJEkddE9YJLMS/JHSb7cPi9Msj3J7va+YOjY65PsSfJUkpVD9UuTPNr23ZIkrX5Kknta/aEkS4barG3fsTvJ2t7nKUk61GzMYD4NPDn0+TpgR1UtBXa0zyS5EFgDXASsAm5NMq+1uQ1YDyxtr1Wtvg7YX1UXAJ8Dbm59LQQ2AJcBy4ANw0EmSeqva8AkWQz8AvDvh8qrgc1tezNwxVD97qp6o6qeBvYAy5KcA5xWVQ9WVQF3Tmkz2ddWYHmb3awEtlfVRFXtB7ZzMJQkSbOg9wzm3wH/BPjzodrZVfUCQHs/q9UXAc8NHbe31Ra17an1Q9pU1QHgFeCMafqSJM2SbgGT5G8D+6pq10ybjKjVNPWjbTM8xvVJdibZ+dJLL81wmJKkmeg5g/kI8EtJngHuBj6a5D8AL7bLXrT3fe34vcC5Q+0XA8+3+uIR9UPaJJkPnA5MTNPXIapqY1WNV9X42NjY0Z+pJOktugVMVV1fVYuragmDxfsHquoTwDZg8q6utcC9bXsbsKbdGXY+g8X8h9tltNeSXN7WV66e0mayryvbdxRwP7AiyYK2uL+i1SRJs2T+HHznZ4AtSdYBzwJXAVTV40m2AE8AB4Brq+rN1uYa4A7gVOC+9gK4HbgryR4GM5c1ra+JJDcCj7Tjbqiqid4nJkk6aFYCpqq+Cny1bb8MLD/McTcBN42o7wQuHlF/nRZQI/ZtAjYd7ZglST8af8kvSerCgJEkdWHASJK6MGAkSV0YMJKkLgwYSVIXBowkqQsDRpLUhQEjSerCgJEkdWHASJK6MGAkSV0YMJKkLgwYSVIXBowkqQsDRpLUhQEjSerCgJEkdWHASJK6mFHAJNkxk5okSZPmT7czyXuA9wJnJlkApO06DfjxzmOTJL2DTRswwN8HfoNBmOziYMC8Cnyx37AkSe900wZMVX0e+HyST1XVF2ZpTJKkE8DbzWAAqKovJPkZYMlwm6q6s9O4JEnvcDMKmCR3AT8JfAt4s5ULMGAkSSPNKGCAceDCqqqeg5EknThm+juYx4C/2HMgkqQTy0xnMGcCTyR5GHhjslhVv9RlVJKkd7yZBsy/ONKO229ovgac0r5na1VtSLIQuIfBDQPPAB+vqv2tzfXAOgbrPP+oqu5v9UuBO4BTga8An66qSnIKg3WgS4GXgV+pqmdam7XAb7Xh/Muq2nyk5yBJOnozvYvsvx9F328AH62q7yd5N/D1JPcBHwN2VNVnklwHXAf80yQXAmuAixj87ua/JfnLVfUmcBuwHvgGg4BZBdzHIIz2V9UFSdYANwO/0kJsA4O1owJ2Jdk2GWSSpP5m+qiY15K82l6vJ3kzyavTtamB77eP726vAlYDk7OJzcAVbXs1cHdVvVFVTwN7gGVJzgFOq6oH200Gd05pM9nXVmB5kgArge1VNdFCZTuDUJIkzZKZzmDeP/w5yRXAsrdrl2QegycAXAB8saoeSnJ2Vb3Q+n0hyVnt8EUMZiiT9rbaD9r21Ppkm+daXweSvAKcMVwf0WZ4fOsZzIw477zz3u50JElH4KieplxV/wX46AyOe7OqLgEWM5iNXDzN4RlRq2nqR9tmeHwbq2q8qsbHxsamGZok6UjN9IeWHxv6+C4Orm3MSFV9L8lXGVymejHJOW32cg6wrx22Fzh3qNli4PlWXzyiPtxmb5L5wOnARKv/3JQ2X53peCVJP7qZzmB+cei1EniNwfrHYSUZS/KBtn0q8DeBbwPbgLXtsLXAvW17G7AmySlJzgeWAg+3y2mvJbm8ra9cPaXNZF9XAg+0dZr7gRVJFrSnQK9oNUnSLJnpGswnj6Lvc4DNbR3mXcCWqvpykgeBLUnWAc8CV7XveDzJFuAJ4ABwbbuDDOAaDt6mfF97AdwO3JVkD4OZy5rW10SSG4FH2nE3VNXEUZyDJOkozfQS2WLgC8BHGFwa+zqD36LsPVybqvpfwIdH1F8Glh+mzU3ATSPqO4G3rN9U1eu0gBqxbxOw6XDjkyT1NdNLZF9icDnqxxncjfX7rSZJ0kgzDZixqvpSVR1orzsAb7uSJB3WTAPmu0k+kWRee32CwaNZJEkaaaYB83eBjwN/CrzA4I6to1n4lySdJGb6sMsbgbVDD6VcCHyWQfBIkvQWM53B/NTwgyLbLb9vuUNMkqRJMw2Yd7UfLAI/nMHMdPYjSToJzTQk/g3wP5NsZfA7mI8z4vcqkiRNmukv+e9MspPBAy4DfKyqnug6MknSO9qML3O1QDFUJEkzclSP65ck6e0YMJKkLgwYSVIXBowkqQsDRpLUhQEjSerCgJEkdWHASJK6MGAkSV0YMJKkLgwYSVIXBowkqQsDRpLUhQEjSerCgJEkdWHASJK6MGAkSV10C5gk5yb5wyRPJnk8yadbfWGS7Ul2t/cFQ22uT7InyVNJVg7VL03yaNt3S5K0+ilJ7mn1h5IsGWqztn3H7iRre52nJGm0njOYA8A/rqq/ClwOXJvkQuA6YEdVLQV2tM+0fWuAi4BVwK1J5rW+bgPWA0vba1WrrwP2V9UFwOeAm1tfC4ENwGXAMmDDcJBJkvrrFjBV9UJVfbNtvwY8CSwCVgOb22GbgSva9mrg7qp6o6qeBvYAy5KcA5xWVQ9WVQF3Tmkz2ddWYHmb3awEtlfVRFXtB7ZzMJQkSbNgVtZg2qWrDwMPAWdX1QswCCHgrHbYIuC5oWZ7W21R255aP6RNVR0AXgHOmKYvSdIs6R4wSd4H/GfgN6rq1ekOHVGraepH22Z4bOuT7Eyy86WXXppmaJKkI9U1YJK8m0G4/HZV/W4rv9gue9He97X6XuDcoeaLgedbffGI+iFtkswHTgcmpunrEFW1sarGq2p8bGzsaE9TkjRCz7vIAtwOPFlV/3Zo1zZg8q6utcC9Q/U17c6w8xks5j/cLqO9luTy1ufVU9pM9nUl8EBbp7kfWJFkQVvcX9FqkqRZMr9j3x8B/g7waJJvtdo/Az4DbEmyDngWuAqgqh5PsgV4gsEdaNdW1Zut3TXAHcCpwH3tBYMAuyvJHgYzlzWtr4kkNwKPtONuqKqJTucpSRqhW8BU1dcZvRYCsPwwbW4CbhpR3wlcPKL+Oi2gRuzbBGya6XglSceWv+SXJHVhwEiSujBgJEldGDCSpC4MGElSFwaMJKkLA0aS1IUBI0nqwoCRJHVhwEiSujBgJEldGDCSpC4MGElSFwaMJKkLA0aS1IUBI0nqwoCRJHVhwEiSujBgJEldGDCSpC4MGElSFwaMJKkLA0aS1IUBI0nqwoCRJHVhwEiSujBgJElddAuYJJuS7Evy2FBtYZLtSXa39wVD+65PsifJU0lWDtUvTfJo23dLkrT6KUnuafWHkiwZarO2fcfuJGt7naMk6fB6zmDuAFZNqV0H7KiqpcCO9pkkFwJrgItam1uTzGttbgPWA0vba7LPdcD+qroA+Bxwc+trIbABuAxYBmwYDjJJ0uzoFjBV9TVgYkp5NbC5bW8Grhiq311Vb1TV08AeYFmSc4DTqurBqirgziltJvvaCixvs5uVwPaqmqiq/cB23hp0kqTOZnsN5uyqegGgvZ/V6ouA54aO29tqi9r21PohbarqAPAKcMY0fUmSZtHxssifEbWapn60bQ790mR9kp1Jdr700kszGqgkaWZmO2BebJe9aO/7Wn0vcO7QcYuB51t98Yj6IW2SzAdOZ3BJ7nB9vUVVbayq8aoaHxsb+xFOS5I01WwHzDZg8q6utcC9Q/U17c6w8xks5j/cLqO9luTytr5y9ZQ2k31dCTzQ1mnuB1YkWdAW91e0miRpFs3v1XGS3wF+DjgzyV4Gd3Z9BtiSZB3wLHAVQFU9nmQL8ARwALi2qt5sXV3D4I60U4H72gvgduCuJHsYzFzWtL4mktwIPNKOu6Gqpt5sIEnqrFvAVNWvHmbX8sMcfxNw04j6TuDiEfXXaQE1Yt8mYNOMBytJOuaOl0V+SdIJxoCRJHVhwEiSujBgJEldGDCSpC4MGElSFwaMJKkLA0aS1IUBI0nqwoCRJHVhwEiSujBgJEldGDCSpC4MGElSFwaMJKkLA0aS1IUBI0nqwoCRJHVhwEiSujBgJEldGDCSpC4MGElSFwaMJKmL+XM9AEmz49kb/tpcD0HHofP++aPd+nYGI0nqwoCRJHVhwEiSujihAybJqiRPJdmT5Lq5Ho8knUxO2IBJMg/4IvC3gAuBX01y4dyOSpJOHidswADLgD1V9Z2q+jPgbmD1HI9Jkk4aJ3LALAKeG/q8t9UkSbPgRP4dTEbU6pADkvXA+vbx+0me6j6qk8eZwHfnehDHg3x27VwPQW/lv89JG0b9r/KI/MThdpzIAbMXOHfo82Lg+eEDqmojsHE2B3WySLKzqsbnehzSKP77nB0n8iWyR4ClSc5P8mPAGmDbHI9Jkk4aJ+wMpqoOJPmHwP3APGBTVT0+x8OSpJPGCRswAFX1FeArcz2Ok5SXHnU889/nLEhVvf1RkiQdoRN5DUaSNIcMGB1zPqJHx6Mkm5LsS/LYXI/lZGHA6JjyET06jt0BrJrrQZxMDBgdaz6iR8elqvoaMDHX4ziZGDA61nxEjyTAgNGx97aP6JF0cjBgdKy97SN6JJ0cDBgdaz6iRxJgwOgYq6oDwOQjep4EtviIHh0PkvwO8CDwwSR7k6yb6zGd6PwlvySpC2cwkqQuDBhJUhcGjCSpCwNGktSFASNJ6sKAkWZJkg8k+fVZ+J4rfMCojgcGjDR7PgDMOGAycDT/jV7B4EnW0pzydzDSLEky+WTpp4A/BH4KWAC8G/itqro3yRLgvrb/pxmExdXArzF4iOh3gV1V9dkkP8ngTyOMAf8X+HvAQuDLwCvt9ctV9b9n6RSlQ8yf6wFIJ5HrgIur6pIk84H3VtWrSc4EvpFk8pE6HwQ+WVW/nmQc+GXgwwz+e/0msKsdtxH4B1W1O8llwK1V9dHWz5erautsnpw0lQEjzY0A/yrJzwJ/zuBPGpzd9v1JVX2jbf8N4N6q+n8ASX6/vb8P+BngPyU/fID1KbM0dmlGDBhpbvwag0tbl1bVD5I8A7yn7fs/Q8eN+vMHMFg//V5VXdJthNKPyEV+afa8Bry/bZ8O7Gvh8vPATxymzdeBX0zynjZr+QWAqnoVeDrJVfDDGwI+NOJ7pDljwEizpKpeBv5HkseAS4DxJDsZzGa+fZg2jzD4cwd/DPwusJPB4j2t3bokfww8zsE/TX038JtJ/qjdCCDNCe8ik45zSd5XVd9P8l7ga8D6qvrmXI9LejuuwUjHv43th5PvATYbLnqncAYjSerCNRhJUhcGjCSpCwNGktSFASNJ6sKAkSR1YcBIkrr4/23/kKhE8JWAAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data['target'])   "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Countplot shows that the target variable values are imbalanced."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## DATA PRE-PROCESSING"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "ID is irrevelant to the given prediction task, so can be removed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "data.drop('id',axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Since the null values are assigned the value -1, they are first replaced to NaN form and then imputed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "data.replace(-1,np.NaN,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Enumerating the null values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "target                 0\n",
       "ps_ind_01              0\n",
       "ps_ind_02_cat        216\n",
       "ps_ind_03              0\n",
       "ps_ind_04_cat         83\n",
       "ps_ind_05_cat       5809\n",
       "ps_ind_06_bin          0\n",
       "ps_ind_07_bin          0\n",
       "ps_ind_08_bin          0\n",
       "ps_ind_09_bin          0\n",
       "ps_ind_10_bin          0\n",
       "ps_ind_11_bin          0\n",
       "ps_ind_12_bin          0\n",
       "ps_ind_13_bin          0\n",
       "ps_ind_14              0\n",
       "ps_ind_15              0\n",
       "ps_ind_16_bin          0\n",
       "ps_ind_17_bin          0\n",
       "ps_ind_18_bin          0\n",
       "ps_reg_01              0\n",
       "ps_reg_02              0\n",
       "ps_reg_03         107772\n",
       "ps_car_01_cat        107\n",
       "ps_car_02_cat          5\n",
       "ps_car_03_cat     411231\n",
       "ps_car_04_cat          0\n",
       "ps_car_05_cat     266551\n",
       "ps_car_06_cat          0\n",
       "ps_car_07_cat      11489\n",
       "ps_car_08_cat          0\n",
       "ps_car_09_cat        569\n",
       "ps_car_10_cat          0\n",
       "ps_car_11_cat          0\n",
       "ps_car_11              5\n",
       "ps_car_12              1\n",
       "ps_car_13              0\n",
       "ps_car_14          42620\n",
       "ps_car_15              0\n",
       "ps_calc_01             0\n",
       "ps_calc_02             0\n",
       "ps_calc_03             0\n",
       "ps_calc_04             0\n",
       "ps_calc_05             0\n",
       "ps_calc_06             0\n",
       "ps_calc_07             0\n",
       "ps_calc_08             0\n",
       "ps_calc_09             0\n",
       "ps_calc_10             0\n",
       "ps_calc_11             0\n",
       "ps_calc_12             0\n",
       "ps_calc_13             0\n",
       "ps_calc_14             0\n",
       "ps_calc_15_bin         0\n",
       "ps_calc_16_bin         0\n",
       "ps_calc_17_bin         0\n",
       "ps_calc_18_bin         0\n",
       "ps_calc_19_bin         0\n",
       "ps_calc_20_bin         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# splitting the data\n",
    "x=data.iloc[:,1:]\n",
    "y=data.iloc[:,0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ps_ind_01</th>\n",
       "      <th>ps_ind_02_cat</th>\n",
       "      <th>ps_ind_03</th>\n",
       "      <th>ps_ind_04_cat</th>\n",
       "      <th>ps_ind_05_cat</th>\n",
       "      <th>ps_ind_06_bin</th>\n",
       "      <th>ps_ind_07_bin</th>\n",
       "      <th>ps_ind_08_bin</th>\n",
       "      <th>ps_ind_09_bin</th>\n",
       "      <th>ps_ind_10_bin</th>\n",
       "      <th>...</th>\n",
       "      <th>ps_calc_11</th>\n",
       "      <th>ps_calc_12</th>\n",
       "      <th>ps_calc_13</th>\n",
       "      <th>ps_calc_14</th>\n",
       "      <th>ps_calc_15_bin</th>\n",
       "      <th>ps_calc_16_bin</th>\n",
       "      <th>ps_calc_17_bin</th>\n",
       "      <th>ps_calc_18_bin</th>\n",
       "      <th>ps_calc_19_bin</th>\n",
       "      <th>ps_calc_20_bin</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "      <td>2.0</td>\n",
       "      <td>5</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1.0</td>\n",
       "      <td>7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5</td>\n",
       "      <td>4.0</td>\n",
       "      <td>9</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>7</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 57 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   ps_ind_01  ps_ind_02_cat  ps_ind_03  ps_ind_04_cat  ps_ind_05_cat  \\\n",
       "0          2            2.0          5            1.0            0.0   \n",
       "1          1            1.0          7            0.0            0.0   \n",
       "2          5            4.0          9            1.0            0.0   \n",
       "3          0            1.0          2            0.0            0.0   \n",
       "4          0            2.0          0            1.0            0.0   \n",
       "\n",
       "   ps_ind_06_bin  ps_ind_07_bin  ps_ind_08_bin  ps_ind_09_bin  ps_ind_10_bin  \\\n",
       "0              0              1              0              0              0   \n",
       "1              0              0              1              0              0   \n",
       "2              0              0              1              0              0   \n",
       "3              1              0              0              0              0   \n",
       "4              1              0              0              0              0   \n",
       "\n",
       "   ...  ps_calc_11  ps_calc_12  ps_calc_13  ps_calc_14  ps_calc_15_bin  \\\n",
       "0  ...           9           1           5           8               0   \n",
       "1  ...           3           1           1           9               0   \n",
       "2  ...           4           2           7           7               0   \n",
       "3  ...           2           2           4           9               0   \n",
       "4  ...           3           1           1           3               0   \n",
       "\n",
       "   ps_calc_16_bin  ps_calc_17_bin  ps_calc_18_bin  ps_calc_19_bin  \\\n",
       "0               1               1               0               0   \n",
       "1               1               1               0               1   \n",
       "2               1               1               0               1   \n",
       "3               0               0               0               0   \n",
       "4               0               0               1               1   \n",
       "\n",
       "   ps_calc_20_bin  \n",
       "0               1  \n",
       "1               0  \n",
       "2               0  \n",
       "3               0  \n",
       "4               0  \n",
       "\n",
       "[5 rows x 57 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    0\n",
       "1    0\n",
       "2    0\n",
       "3    0\n",
       "4    0\n",
       "Name: target, dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Feature elimination need to be carried out as the data contain considerably large number of predictor variables, so as to reduce the computational complexity."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Checking for Multi-collinearity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "continuous=[]\n",
    "categorical=[]\n",
    "binary=[]\n",
    "ordinal=[]\n",
    "for i in x.columns:\n",
    "    if 'cat' in i:\n",
    "        categorical.append(i)\n",
    "    elif 'bin' in i:\n",
    "        binary.append(i)\n",
    "    elif x[i].dtype=='float64':\n",
    "        continuous.append(i)\n",
    "    elif x[i].dtype=='int64':\n",
    "        ordinal.append(i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x576 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,8))\n",
    "corr_continuous=x[continuous].corr()\n",
    "sns.heatmap(corr_continuous,square=True,annot=True,cmap='YlGnBu')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Among the continuous variables,ps_reg_02 and ps_reg_03 have a strong postive correlation, so either of them can be removed.The same applies for ps_car_12 and ps_car_13."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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M6Itt1oWBLQcJOmrJ8annbBSWv5m17QTVoo/NyXtBClCgxwRERERkVrqp/pGgg5XExERERGRWkV7ckREREREaimzn+a+HO3JERERERGRWkWDHBERERERqVV0uJqIiIiISK2lw9VERERERETC054cEREREZFa6qe5H0d7ckREREREpJbRIEdERERERGoVHa4mIiIiIlJr/TQPWNOeHBERERERqVW0J0dEREREpJYy054cERERERGR8DTIERERERGRWkWDHBERERERqVU0yBERERERkVpFJx4owt3pf9NdjH/lDdZouAaXXnsRHbfYfIXfmzZ1OtdcfB1zvp/L5ltsxhXXX0L9+vWLTv/Gq2/S/6a7WLZsGQcffhAnnPpbAD795DP6XX8bi35cRN16dTnv0nPZcutOLF68mJt73conH/2LOnWMc3r+ge1/vl3B9kK3X8pjKzTtnO/ncPVFvZg+bQYbtmrBNX2vZO111mbC629xT/8hLF68hPr163Hmeaez487bl928HXLXn3ll7KvUsTqsu966XHbtRTTboFneebtdNfP2jpzbP77AvH0jp23znLZ807707DiGDRzOpM+/ZOD9d9Fpq44AfDzxE27udUvmdnFOOeNk9u6yZ9F5m1brhNffYlCV5WCHapaD3IaaXBY+++e/6Xf9rcyfv5ANW7XgTzdcxlqN1+LZvz3PqOEPVd7nvz/9D0MeGMhmnTqk1lponZCvdXBOa1U1uSxUPKf+mX1O9ahmfZXWvH3p2XH8Odt/T05/dctBrncnvMcdfe9myZIlNGnahDvuvTXvYxty51Beem4cderW5bCuh3DUcUcUnR+5Cq17p0/9ihOP+B1t2m4MwJbbbMGFV5xX9LbKff1Vk9uuCjOmz+CkI07llDNO5rcnHw3A80+/yF/uHYkZNGvejCuuv5R1mzZJtbfY64BSe0tty7Wyz7kZX83khitu5JtvZlPHjEOOPJiuxx8JFN4OJ+XeC27m17v8gpnffc3W3X+R2P1IedKenCLGv/ImU76cwsgn7qPnn87nlutvz/t799w2mKNPOJIHnryPtddpzN8efbro9EuXLuXW3v3pe1dv7ntkKC+MeZEv/v0FAANuG8Qpp5/I0IcGceqZpzDwtkEAPPnXvwEw/OEh3DLwJu66ZSDLli3L21Ps9qt7bMWmHTH0AXbYZQceePI+dthl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\n",
      "text/plain": [
       "<Figure size 1080x864 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(15,12))\n",
    "corr_ordinal=x[ordinal].corr()\n",
    "sns.heatmap(corr_ordinal,square=True,annot=True,cmap='Greens')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "No strong correlations are observed in the Ordinal variables."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Checking feature importances"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The contribution of each of the predictor variables to the prediction can be evaluated using Recursive feature elimination technique(RFE),statistical methods or the in_built decision tree method of XGBoost classifier. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[11:37:36] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
       "              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,\n",
       "              importance_type='gain', interaction_constraints='',\n",
       "              learning_rate=0.300000012, max_delta_step=0, max_depth=6,\n",
       "              min_child_weight=1, missing=nan, monotone_constraints='()',\n",
       "              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,\n",
       "              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,\n",
       "              tree_method='exact', validate_parameters=1, verbosity=None)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "importance_model=XGBClassifier()\n",
    "importance_model.fit(x,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "imp=importance_model.feature_importances_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<BarContainer object of 57 artists>"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,8))\n",
    "plt.bar(range(len(imp)),imp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>IMPORTANCE</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>ps_ind_01</th>\n",
       "      <td>0.019118</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_02_cat</th>\n",
       "      <td>0.020086</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_03</th>\n",
       "      <td>0.019646</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_04_cat</th>\n",
       "      <td>0.018457</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_05_cat</th>\n",
       "      <td>0.041839</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_06_bin</th>\n",
       "      <td>0.026932</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_07_bin</th>\n",
       "      <td>0.025347</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_08_bin</th>\n",
       "      <td>0.025555</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_09_bin</th>\n",
       "      <td>0.024807</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_10_bin</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_11_bin</th>\n",
       "      <td>0.012140</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_12_bin</th>\n",
       "      <td>0.020424</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_13_bin</th>\n",
       "      <td>0.011175</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_14</th>\n",
       "      <td>0.008883</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_15</th>\n",
       "      <td>0.018732</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_16_bin</th>\n",
       "      <td>0.026633</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_17_bin</th>\n",
       "      <td>0.029036</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_ind_18_bin</th>\n",
       "      <td>0.014039</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_reg_01</th>\n",
       "      <td>0.020322</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_reg_02</th>\n",
       "      <td>0.018200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_reg_03</th>\n",
       "      <td>0.018055</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_01_cat</th>\n",
       "      <td>0.019809</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_02_cat</th>\n",
       "      <td>0.014988</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_03_cat</th>\n",
       "      <td>0.028611</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_04_cat</th>\n",
       "      <td>0.015902</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_05_cat</th>\n",
       "      <td>0.014106</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_06_cat</th>\n",
       "      <td>0.015604</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_07_cat</th>\n",
       "      <td>0.022181</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_08_cat</th>\n",
       "      <td>0.018939</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_09_cat</th>\n",
       "      <td>0.019598</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_10_cat</th>\n",
       "      <td>0.012910</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_11_cat</th>\n",
       "      <td>0.015596</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_11</th>\n",
       "      <td>0.017913</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_12</th>\n",
       "      <td>0.014215</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_13</th>\n",
       "      <td>0.023640</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_14</th>\n",
       "      <td>0.016585</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_car_15</th>\n",
       "      <td>0.016599</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_01</th>\n",
       "      <td>0.014831</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_02</th>\n",
       "      <td>0.014489</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_03</th>\n",
       "      <td>0.013619</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_04</th>\n",
       "      <td>0.013640</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_05</th>\n",
       "      <td>0.014311</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_06</th>\n",
       "      <td>0.014421</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_07</th>\n",
       "      <td>0.014010</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_08</th>\n",
       "      <td>0.015010</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_09</th>\n",
       "      <td>0.015416</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_10</th>\n",
       "      <td>0.013728</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_11</th>\n",
       "      <td>0.014045</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_12</th>\n",
       "      <td>0.015654</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_13</th>\n",
       "      <td>0.012884</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_14</th>\n",
       "      <td>0.013882</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_15_bin</th>\n",
       "      <td>0.015038</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_16_bin</th>\n",
       "      <td>0.014890</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_17_bin</th>\n",
       "      <td>0.014875</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_18_bin</th>\n",
       "      <td>0.014907</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_19_bin</th>\n",
       "      <td>0.017985</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ps_calc_20_bin</th>\n",
       "      <td>0.015746</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                IMPORTANCE\n",
       "ps_ind_01         0.019118\n",
       "ps_ind_02_cat     0.020086\n",
       "ps_ind_03         0.019646\n",
       "ps_ind_04_cat     0.018457\n",
       "ps_ind_05_cat     0.041839\n",
       "ps_ind_06_bin     0.026932\n",
       "ps_ind_07_bin     0.025347\n",
       "ps_ind_08_bin     0.025555\n",
       "ps_ind_09_bin     0.024807\n",
       "ps_ind_10_bin     0.000000\n",
       "ps_ind_11_bin     0.012140\n",
       "ps_ind_12_bin     0.020424\n",
       "ps_ind_13_bin     0.011175\n",
       "ps_ind_14         0.008883\n",
       "ps_ind_15         0.018732\n",
       "ps_ind_16_bin     0.026633\n",
       "ps_ind_17_bin     0.029036\n",
       "ps_ind_18_bin     0.014039\n",
       "ps_reg_01         0.020322\n",
       "ps_reg_02         0.018200\n",
       "ps_reg_03         0.018055\n",
       "ps_car_01_cat     0.019809\n",
       "ps_car_02_cat     0.014988\n",
       "ps_car_03_cat     0.028611\n",
       "ps_car_04_cat     0.015902\n",
       "ps_car_05_cat     0.014106\n",
       "ps_car_06_cat     0.015604\n",
       "ps_car_07_cat     0.022181\n",
       "ps_car_08_cat     0.018939\n",
       "ps_car_09_cat     0.019598\n",
       "ps_car_10_cat     0.012910\n",
       "ps_car_11_cat     0.015596\n",
       "ps_car_11         0.017913\n",
       "ps_car_12         0.014215\n",
       "ps_car_13         0.023640\n",
       "ps_car_14         0.016585\n",
       "ps_car_15         0.016599\n",
       "ps_calc_01        0.014831\n",
       "ps_calc_02        0.014489\n",
       "ps_calc_03        0.013619\n",
       "ps_calc_04        0.013640\n",
       "ps_calc_05        0.014311\n",
       "ps_calc_06        0.014421\n",
       "ps_calc_07        0.014010\n",
       "ps_calc_08        0.015010\n",
       "ps_calc_09        0.015416\n",
       "ps_calc_10        0.013728\n",
       "ps_calc_11        0.014045\n",
       "ps_calc_12        0.015654\n",
       "ps_calc_13        0.012884\n",
       "ps_calc_14        0.013882\n",
       "ps_calc_15_bin    0.015038\n",
       "ps_calc_16_bin    0.014890\n",
       "ps_calc_17_bin    0.014875\n",
       "ps_calc_18_bin    0.014907\n",
       "ps_calc_19_bin    0.017985\n",
       "ps_calc_20_bin    0.015746"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "importances=pd.DataFrame(imp,index=x.columns,columns=['IMPORTANCE'])\n",
    "importances"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From the above feature importances, it is obvious that all the features contribute considerably to the prediction, therefore principle component analysis for dimensionality reduction will not be an appropriate option.However inorder to avoid multicollinearity we can eliminate one of the two strongly correlated variables taking their importances in to account.\n",
    "\n",
    "Among 'ps_reg_02' and 'ps_reg_03', the latter has less importance, so can be removed.\n",
    "\n",
    "Among 'ps_car_12' and 'ps_car_13', the former has less importance, so can be removed.\n",
    "\n",
    "Also the variable 'ps_ind_10_bin' does not seem to contribute to the prediction, so can be eliminated."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "x.drop(['ps_reg_03','ps_car_12','ps_ind_10_bin'],axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Dealing with the null values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mean_impute=SimpleImputer(strategy='mean',missing_values=np.NaN)\n",
    "mode_impute=SimpleImputer(strategy='most_frequent',missing_values=np.NaN)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "# A loop is made for imputing categorical variables with mode and continuous values with mean\n",
    "for i in x.columns:\n",
    "    if x[i].isnull().sum()>0:\n",
    "        if 'cat' in i:\n",
    "            x[i]=mode_impute.fit_transform(np.array(x[i]).reshape(-1,1))\n",
    "        elif 'bin' in i:\n",
    "            x[i]=mode_impute.fit_transform(np.array(x[i]).reshape(-1,1))\n",
    "        elif x[i].dtype=='float64':\n",
    "            x[i]=mean_impute.fit_transform(np.array(x[i]).reshape(-1,1))\n",
    "        elif x[i].dtype=='int64':\n",
    "            x[i]=mode_impute.fit_transform(np.array(x[i]).reshape(-1,1))\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ps_ind_01         0\n",
       "ps_ind_02_cat     0\n",
       "ps_ind_03         0\n",
       "ps_ind_04_cat     0\n",
       "ps_ind_05_cat     0\n",
       "ps_ind_06_bin     0\n",
       "ps_ind_07_bin     0\n",
       "ps_ind_08_bin     0\n",
       "ps_ind_09_bin     0\n",
       "ps_ind_11_bin     0\n",
       "ps_ind_12_bin     0\n",
       "ps_ind_13_bin     0\n",
       "ps_ind_14         0\n",
       "ps_ind_15         0\n",
       "ps_ind_16_bin     0\n",
       "ps_ind_17_bin     0\n",
       "ps_ind_18_bin     0\n",
       "ps_reg_01         0\n",
       "ps_reg_02         0\n",
       "ps_car_01_cat     0\n",
       "ps_car_02_cat     0\n",
       "ps_car_03_cat     0\n",
       "ps_car_04_cat     0\n",
       "ps_car_05_cat     0\n",
       "ps_car_06_cat     0\n",
       "ps_car_07_cat     0\n",
       "ps_car_08_cat     0\n",
       "ps_car_09_cat     0\n",
       "ps_car_10_cat     0\n",
       "ps_car_11_cat     0\n",
       "ps_car_11         0\n",
       "ps_car_13         0\n",
       "ps_car_14         0\n",
       "ps_car_15         0\n",
       "ps_calc_01        0\n",
       "ps_calc_02        0\n",
       "ps_calc_03        0\n",
       "ps_calc_04        0\n",
       "ps_calc_05        0\n",
       "ps_calc_06        0\n",
       "ps_calc_07        0\n",
       "ps_calc_08        0\n",
       "ps_calc_09        0\n",
       "ps_calc_10        0\n",
       "ps_calc_11        0\n",
       "ps_calc_12        0\n",
       "ps_calc_13        0\n",
       "ps_calc_14        0\n",
       "ps_calc_15_bin    0\n",
       "ps_calc_16_bin    0\n",
       "ps_calc_17_bin    0\n",
       "ps_calc_18_bin    0\n",
       "ps_calc_19_bin    0\n",
       "ps_calc_20_bin    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Normalizing the continuous variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler=StandardScaler()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in x.columns:\n",
    "    if 'cat' not in i or 'bin' not in i: \n",
    "        if x[i].dtype=='float64':\n",
    "            x[i]=scaler.fit_transform(np.array(x[i]).reshape(-1,1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Splitting the dataset into train and test data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Synthetic data generation to balance the target variable values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Since the target variable values are imbalanced and it can cause bias in the model,we genrate synthetic data using smote function from imblearn api."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "oversampler=SMOTE()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,y_train=oversampler.fit_resample(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='target', ylabel='count'>"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Building the machine learning model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To find out the best model for this particular classification problem all the important machine learning models are tried out and to find out the one which is more effective an Receiver operating charecteristic(ROC) curve and corresponding area under the curve(AUC) are considered for each model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "logr=LogisticRegression()\n",
    "knn=KNeighborsClassifier(n_neighbors=3)\n",
    "forest=RandomForestClassifier(max_depth=4)\n",
    "adab=AdaBoostClassifier()\n",
    "xgb=XGBClassifier(max_depth=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logr.fit(x_train,y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier(n_neighbors=3)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "knn.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(max_depth=4)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "forest.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "AdaBoostClassifier()"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "adab.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[09:42:29] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
       "              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,\n",
       "              importance_type='gain', interaction_constraints='',\n",
       "              learning_rate=0.300000012, max_delta_step=0, max_depth=4,\n",
       "              min_child_weight=1, missing=nan, monotone_constraints='()',\n",
       "              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,\n",
       "              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,\n",
       "              tree_method='exact', validate_parameters=1, verbosity=None)"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xgb.fit(x_train,y_train,verbose=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_list=[logr,knn,forest,adab,xgb]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "colors=['r','g','b','y','m']\n",
    "labels=['logistic regression','K nearest neighbour ','Random forest','Adaboost','Xgboost']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy score for LogisticRegression() model is\n",
      "0.8679888779684651\n",
      "accuracy score for KNeighborsClassifier(n_neighbors=3) model is\n",
      "0.6928420822727922\n",
      "accuracy score for RandomForestClassifier(max_depth=4) model is\n",
      "0.8251976176675655\n",
      "accuracy score for AdaBoostClassifier() model is\n",
      "0.9039758742639215\n",
      "accuracy score for XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
      "              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,\n",
      "              importance_type='gain', interaction_constraints='',\n",
      "              learning_rate=0.300000012, max_delta_step=0, max_depth=4,\n",
      "              min_child_weight=1, missing=nan, monotone_constraints='()',\n",
      "              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,\n",
      "              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,\n",
      "              tree_method='exact', validate_parameters=1, verbosity=None) model is\n",
      "0.9631645707853465\n"
     ]
    }
   ],
   "source": [
    "for i in model_list:\n",
    "    print('accuracy score for',i,'model is')\n",
    "    print(accuracy_score(y_true=y_test,y_pred=i.predict(x_test.values)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x1bdbf4e57f0>"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "a=0\n",
    "b=0\n",
    "plt.figure(figsize=(10,8))\n",
    "for i in model_list:\n",
    "    fpr,tpr,thresholds=roc_curve(y_test,i.predict_proba(x_test.values)[:,1])\n",
    "    plt.plot(fpr,tpr,color=colors[a],label=labels[b])\n",
    "    a=a+1\n",
    "    b=b+1\n",
    "plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')\n",
    "plt.title('Receiver Operating Characteristic (ROC) Curve')\n",
    "plt.legend()\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "auc for LogisticRegression() model is\n",
      "0.5200065882309699\n",
      "auc for KNeighborsClassifier(n_neighbors=3) model is\n",
      "0.5204316409080771\n",
      "auc for RandomForestClassifier(max_depth=4) model is\n",
      "0.5514026284861671\n",
      "auc for AdaBoostClassifier() model is\n",
      "0.529714982236698\n",
      "auc for XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
      "              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,\n",
      "              importance_type='gain', interaction_constraints='',\n",
      "              learning_rate=0.300000012, max_delta_step=0, max_depth=4,\n",
      "              min_child_weight=1, missing=nan, monotone_constraints='()',\n",
      "              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,\n",
      "              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,\n",
      "              tree_method='exact', validate_parameters=1, verbosity=None) model is\n",
      "0.5955971765758674\n"
     ]
    }
   ],
   "source": [
    "for i in model_list:\n",
    "    print('auc for',i,'model is')\n",
    "    print(roc_auc_score(y_test,i.predict_proba(x_test)[:,1]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Hyper-parameter tuning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From the above results it can be concluded tht among the various classification model the best one whic suits the problem in hand is the XGBoost classifier model. Inorder to further improve the accuracy hyperparameter tuning is done to find out the best combination of parameters for the XGBoost model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "parameters={'max_depth':[2,3,4,5],'n_estimators':[200,500,750,1000,1500,2000],'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6]}\n",
    "model=XGBClassifier()\n",
    "grid=RandomizedSearchCV(model,parameters,cv=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[21:17:44] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[21:21:58] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[21:25:52] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[21:29:35] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[21:47:30] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[22:02:56] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[22:18:12] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[22:23:53] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[22:29:29] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[22:35:04] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[22:45:13] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[22:55:27] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[23:06:33] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[23:21:57] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[23:37:10] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[23:52:24] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[00:18:31] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[00:44:01] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[01:09:42] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[01:16:10] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[01:22:32] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[01:28:57] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[01:35:21] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[01:41:47] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[01:48:09] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[01:54:02] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[01:59:58] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[02:05:54] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[02:21:17] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[02:36:44] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "[02:52:17] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "RandomizedSearchCV(cv=3,\n",
       "                   estimator=XGBClassifier(base_score=None, booster=None,\n",
       "                                           colsample_bylevel=None,\n",
       "                                           colsample_bynode=None,\n",
       "                                           colsample_bytree=None, gamma=None,\n",
       "                                           gpu_id=None, importance_type='gain',\n",
       "                                           interaction_constraints=None,\n",
       "                                           learning_rate=None,\n",
       "                                           max_delta_step=None, max_depth=None,\n",
       "                                           min_child_weight=None, missing=nan,\n",
       "                                           monotone_constraints=None,\n",
       "                                           n_estimators=100, n_jobs=None,\n",
       "                                           num_parallel_tree=None,\n",
       "                                           random_state=None, reg_alpha=None,\n",
       "                                           reg_lambda=None,\n",
       "                                           scale_pos_weight=None,\n",
       "                                           subsample=None, tree_method=None,\n",
       "                                           validate_parameters=None,\n",
       "                                           verbosity=None),\n",
       "                   param_distributions={'learning_rate': [0.1, 0.2, 0.3, 0.4,\n",
       "                                                          0.5, 0.6],\n",
       "                                        'max_depth': [2, 3, 4, 5],\n",
       "                                        'n_estimators': [200, 500, 750, 1000,\n",
       "                                                         1500, 2000]})"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid.fit(x,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'n_estimators': 750, 'max_depth': 3, 'learning_rate': 0.1}"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9635558422881259"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid.best_score_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model=XGBClassifier(n_estimators=750,max_depth=3,learning_rate=0.1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[12:08:12] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
       "              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,\n",
       "              importance_type='gain', interaction_constraints='',\n",
       "              learning_rate=0.1, max_delta_step=0, max_depth=3,\n",
       "              min_child_weight=1, missing=nan, monotone_constraints='()',\n",
       "              n_estimators=750, n_jobs=4, num_parallel_tree=1, random_state=0,\n",
       "              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,\n",
       "              tree_method='exact', validate_parameters=1, verbosity=None)"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "best_model.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## SAVING AND TESTING THE BEST MODEL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['insurance_claim_xgb_model']"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "joblib.dump(best_model,'insurance_claim_xgb_model')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "model=joblib.load('insurance_claim_xgb_model')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions=model.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9631645707853465"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_true=y_test,y_pred=predictions)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Project Analysis\n",
    "The object of this project was to build a Machine learning model to predict whether a auto insurance policy holder files a claim or not. The given dataset had 59 features including 'target' and irrevelant features like 'ID'. The size of the data itself was the first issue to be adressed as it can increase the computational complexity.Firstly,The target variable and the predictor variables were split into seperate variables and then features were organized into seperate lists of continuous, binary,categorical(nominal) and ordinal variables. To check multi-collinearity within continuous and ordinal variables, heatmaps(from seaborn package) of the same were plotted to find out that some of the continuous features shown strong correlation.One among two strongly correlated features were eliminated after considering their significance in the prediction using XGBclassifier.\n",
    "\n",
    "The null values in the dataset were assigned the value -1 which were replaced by the standard NaN form and then imputed using the SimpleImputer from scikit learn. The null values in all the categorical variables were replaced with the most frequent(mode) value, while that of continuous variables were replaced with the average(mean value).\n",
    "Afterwards the data was split into training and test data by using the train_test_split function from scikit learn.\n",
    "It was observed that the target variable values were imbalanced in the dataset. Considering that the imbalanced dataset can lead to a biased and inaccurate model, SMOTE package from imblearn API was used for oversampling or synthetic data generation to balance the dataset.\n",
    "\n",
    "Since there is no single perfect model for all the prediction tasks, all the classification algorithms including Logistic regression, Random forest classifier, XGBoost classifier, AdaBoost classifier and K-nearest Neighbour classifer, were tried out and the most effective one was chosen by plotting the receiver operator curves(ROC) and evaluating the Area under the curve values(AUC) for each of the models.\n",
    "XGBclassifier was found to be the most effective in prediction and hyperparameter tuning for the same was carried out using the RandomizedSearchCV function from scikit learn. The best model was saved using the joblib package and was tested on the test data to get an accuracy of ~96.35%\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
