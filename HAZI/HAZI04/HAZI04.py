# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
'''
FONTOS: Az első feladatáltal visszaadott DataFrame-et kell használni a további feladatokhoz. 
A függvényeken belül mindig készíts egy másolatot a bemenő df-ről, (new_df = df.copy() és ezzel dolgozz tovább.)
'''

# %%
'''
Készíts egy függvényt, ami egy string útvonalat vár paraméterként, és egy DataFrame ad visszatérési értékként.

Egy példa a bemenetre: 'test_data.csv'
Egy példa a kimenetre: df_data
return type: pandas.core.frame.DataFrame
függvény neve: csv_to_df
'''

# %%
def csv_to_df(input: str)->pd.DataFrame:
    df_data=pd.read_csv(input)
    return df_data

#df=csv_to_df('StudentsPerformance.csv')

# %%
'''
Készíts egy függvényt, ami egy DataFrame-et vár paraméterként, 
és átalakítja azoknak az oszlopoknak a nevét nagybetűsre amelyiknek neve nem tartalmaz 'e' betüt.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_capitalized
return type: pandas.core.frame.DataFrame
függvény neve: capitalize_columns
'''

# %%
def capitalize_columns(df_data: pd.DataFrame):
    df_copy=df_data.copy()
    for item in df_copy.columns:
        if 'e' not in str(item):
            df_copy.rename(columns={item:item.upper()},inplace=True)
    return df_copy

#print(capitalize_columns(df))


# %%
'''
Készíts egy függvényt, ahol egy szám formájában vissza adjuk, hogy hány darab diáknak sikerült teljesíteni a matek vizsgát.
(legyen az átmenő ponthatár 50).

Egy példa a bemenetre: df_data
Egy példa a kimenetre: 5
return type: int
függvény neve: math_passed_count
'''

# %%
def math_passed_count(df_data: pd.DataFrame)->int:
    counter=0
    for item in df_data['math score']:
        if item>=50:
            counter+=1
    return counter

#math_passed_count(df)


# %%
'''
Készíts egy függvényt, ahol Dataframe ként vissza adjuk azoknak a diákoknak az adatait (sorokat), akik végeztek előzetes gyakorló kurzust.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_did_pre_course
return type: pandas.core.frame.DataFrame
függvény neve: did_pre_course
'''

# %%
def did_pre_course(df_data: pd.DataFrame)->pd.DataFrame:
    return df_data[df_data['test preparation course']=='completed']

#did_pre_course(df)

# %%
'''
Készíts egy függvényt, ahol a bemeneti Dataframet a diákok szülei végzettségi szintjei alapján csoportosításra kerül,
majd aggregációként vegyük, hogy átlagosan milyen pontszámot értek el a diákok a vizsgákon.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_average_scores
return type: pandas.core.frame.DataFrame
függvény neve: average_scores
'''

# %%
def average_scores(df_data: pd.DataFrame)->pd.DataFrame:
    df_copy=df_data.copy()
    grouped=df_copy.groupby('parental level of education').aggregate({'math score':'mean','reading score':'mean','writing score':'mean'})
    return grouped

#average_scores(df)

# %%
'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'age' oszloppal, töltsük fel random 18-66 év közötti értékekkel.
A random.randint() függvényt használd, a random sorsolás legyen seedleve, ennek értéke legyen 42.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_age
return type: pandas.core.frame.DataFrame
függvény neve: add_age
'''

# %%
def add_age(df_data: pd.DataFrame)->pd.DataFrame:
    df_copy=df_data.copy()
    np.random.seed(42)
    ages=np.random.randint(18,67,size=len(df_copy))
    df_copy['age']=ages
    return df_copy

#add_age(df)      

# %%
'''
Készíts egy függvényt, ami vissza adja a legjobb teljesítményt elérő női diák pontszámait.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: (99,99,99) #math score, reading score, writing score
return type: tuple
függvény neve: female_top_score
'''

# %%
def female_top_score(df_data:pd.DataFrame)->tuple:
    df_copy = df_data.copy()
    columns = ['math score', 'reading score', 'writing score']
    values = df_copy[df_copy[columns].sum(axis=1) == df_copy[columns].sum(axis=1).max()]
    tmp = values[values['gender']=='female'].iloc[0]
    return (tmp[columns[0]],tmp[columns[1]],tmp[columns[2]])

#female_top_score(df)

# %%
'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'grade' oszloppal. 
Számoljuk ki hogy a diákok hány százalékot ((math+reading+writing)/300) értek el a vizsgán, és osztályozzuk őket az alábbi szempontok szerint:

90-100%: A
80-90%: B
70-80%: C
60-70%: D
<60%: F

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_grade
return type: pandas.core.frame.DataFrame
függvény neve: add_grade
'''

# %%
def add_grade(df_data:pd.DataFrame)->pd.DataFrame:
    df_copy = df_data.copy()
    grades = (df_copy['math score']+df_copy['reading score']+df_copy['writing score'])/300    
    df_copy['grade'] = pd.cut(grades, bins = [0, 0.6,0.7,0.8,0.9,1], right=True, labels=['F','D','C','B','A'])
    return df_copy
#add_grade(df)

# %%
'''
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan oszlop diagrammot,
ami vizualizálja a nemek által elért átlagos matek pontszámot.

Oszlopdiagram címe legyen: 'Average Math Score by Gender'
Az x tengely címe legyen: 'Gender'
Az y tengely címe legyen: 'Math Score'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: math_bar_plot
'''

# %%
def math_bar_plot(df_data:pd.DataFrame)->plt.figure:
    df_copy=df_data.copy()
    group=df_copy.groupby('gender')['math score'].mean()
    fig,ax = plt.subplots() 
    fig=group.plot.bar(title='Average Math Score by Gender',ylabel='Math Score', xlabel='Gender',ax=ax,fig=fig)
    return fig
#math_bar_plot(df)

# %%
''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan histogramot,
ami vizualizálja az elért írásbeli pontszámokat.

A histogram címe legyen: 'Distribution of Writing Scores'
Az x tengely címe legyen: 'Writing Score'
Az y tengely címe legyen: 'Number of Students'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: writing_hist
'''

# %%
def writing_hist(df_data: pd.DataFrame)->plt.figure:
    df_copy=df_data.copy()
    fig, ax = plt.subplots()
    fig=df_copy.plot.hist(column='writing score',xlabel='Writing Score',ylabel='Number of Students',title='Distribution of Writing Scores',ax=ax,fig = fig)
    return fig
#writing_hist(df)

# %%
''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan kördiagramot,
ami vizualizálja a diákok etnikum csoportok szerinti eloszlását százalékosan.

Érdemes megszámolni a diákok számát, etnikum csoportonként,majd a százalékos kirajzolást az autopct='%1.1f%%' paraméterrel megadható.
Mindegyik kör szelethez tartozzon egy címke, ami a csoport nevét tartalmazza.
A diagram címe legyen: 'Proportion of Students by Race/Ethnicity'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: ethnicity_pie_chart
'''

# %%
def ethnicity_pie_chart(df_data:pd.DataFrame)->plt.figure:
    df_copy=df_data.copy()
    group = df_copy.groupby('race/ethnicity')['race/ethnicity'].count()
    fig,ax = plt.subplots()
    fig=group.plot.pie(title='Proportion of Students by Race/Ethnicity',autopct='%1.1f%%',fig = fig,ax=ax)
    return fig
#ethnicity_pie_chart(df)


