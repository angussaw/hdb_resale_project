from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
from tqdm import tqdm_pandas, tqdm
import re
import logging

tqdm_pandas(tqdm())

logger = logging.getLogger(__name__)

def find_coordinates(amenities_df: pd.DataFrame) -> pd.DataFrame:
    """Given the names, get the coordinates of all amenities in a dataframe

    Args:
        amenities_df (pd.DataFrame): dataframe containing the names of the amenities

    Returns:
        pd.DataFrame: dataframe containing
    """

    # Do not need to change the URL
    url = (
        "https://developers.onemap.sg/commonapi/search?returnGeom=Y&getAddrDetails=Y&pageNum=1&searchVal="
    )

    output_df = pd.DataFrame()
    found = 0
    for add in tqdm(list(amenities_df["Name"])):
        add_url = (url+ add)
        # Retrieve information from website
        response = requests.get(add_url)
        try:
            data = json.loads(response.text)
        except ValueError:
            print("JSONDecodeError")
            pass

        if len(data["results"]) != 0:
            result = pd.DataFrame([data["results"][0]])
            result["address"] = add

            output_df = pd.concat([output_df,result])
            found += 1

    print(f"Found coordinates for {found} out of {len(amenities_df)} amenities")

    return output_df

#################
#### SCHOOLS ####
#################

# primary schools
prisch="https://en.wikipedia.org/wiki/List_of_primary_schools_in_Singapore"
response = requests.get(prisch)
print(f"Primary schools: {response.status_code}")

soup = BeautifulSoup(response.text, 'html.parser')
prischooltable = soup.find('table',{'class':"wikitable"})
prischooldf = pd.DataFrame(pd.read_html(str(prischooltable))[0])
prischooldf = prischooldf[["Name"]]

# secondary schools
secsch="https://en.wikipedia.org/wiki/List_of_secondary_schools_in_Singapore"
response = requests.get(secsch)
print(f"Secondary schools: {response.status_code}")
soup = BeautifulSoup(response.text, 'html.parser')
secschooltable = soup.find('table',{'class':"wikitable"})
secschooldf = pd.DataFrame(pd.read_html(str(secschooltable))[0])
secschooldf = secschooldf[["Name"]]

# junior colleges and polytechnics
jcsch="https://en.wikipedia.org/wiki/List_of_schools_in_Singapore"
response=requests.get(jcsch)
print(f"Poly and JCs: {response.status_code}")
soup = BeautifulSoup(response.text, 'html.parser')
polyjcschooltable = soup.find_all('table',{'class':"wikitable"})
polyjcschooldf = pd.read_html(str(polyjcschooltable))

jcschooldf = pd.DataFrame(polyjcschooldf[0])
jcschooldf = jcschooldf["College name"][["English"]]
jcschooldf = jcschooldf.rename(columns={"English":"Name"})

polyschooldf = pd.DataFrame(polyjcschooldf[1])
polyschooldf = polyschooldf["Polytechnic"][["Full name"]]
polyschooldf = polyschooldf.rename(columns={"Full name":"Name"})
polyschooldf["Name"] = polyschooldf["Name"].apply(lambda x: x.split("Polytechnic")[0] + "Polytechnic")

# Universities
unisch=[{"Name":"National University of Singapore"},
       {"Name":"Nanyang Technological University"},
       {"Name":"Singapore Management University"},
       {"Name":"Singapore University of Technology and Design"},
       {"Name":"Singapore Institute of Technology"},
       {"Name":"Singapore University of Social Sciences"}]
unischooldf = pd.DataFrame(unisch)

school_df = pd.concat([prischooldf,secschooldf,jcschooldf,polyschooldf,unischooldf])

print("Retrieving school coordinates.....")
school_coordinates = find_coordinates(school_df)
school_coordinates.to_csv("data/for_feature_engineering/schools/school_coordinates_test.csv", index=False)


######################
#### MRT STATIONS ####
######################
mrt_stations="https://en.wikipedia.org/wiki/List_of_Singapore_MRT_stations"
response = requests.get(mrt_stations)
print(f"MRT stations: {response.status_code}")
soup = BeautifulSoup(response.text, 'html.parser')
mrt_stations_table = soup.find_all('table',{'class':"wikitable"})
mrt_stations_table = pd.DataFrame(pd.read_html(str(mrt_stations_table))[2])

months = {"Jan":"01","Feb":"02","Mar":"03","Apr":"04","May":"05","Jun":"06","Jul":"07","Aug":"08","Sep":"09","Oct":"10","Nov":"11","Dec":"12"}
mrt_stations_df = pd.concat([mrt_stations_table["Station name"][["English • Malay"]], mrt_stations_table["Opening"]], axis = "columns")
mrt_stations_df.columns = ["Name","Opening date"]
mrt_stations_df = mrt_stations_df[mrt_stations_df["Name"] != mrt_stations_df["Opening date"]]
mrt_stations_df["Name"] = mrt_stations_df["Name"].apply(lambda x: x.replace(' • Kebun Bunga',''))
mrt_stations_df["Name"] = mrt_stations_df["Name"].apply(lambda x: x.upper() + " MRT STATION")
mrt_stations_df = mrt_stations_df[mrt_stations_df["Opening date"].apply(lambda x: len(x) >= 10)]
mrt_stations_df["Opening date"] = mrt_stations_df["Opening date"].apply(lambda x: x.replace('[19]',''))
mrt_stations_df["Opening month"] = mrt_stations_df["Opening date"].apply(lambda x: x.split(" ")[1][0:3]).map(months)
mrt_stations_df["Opening month"] = mrt_stations_df["Opening month"].astype(int)
mrt_stations_df["Opening year"] = mrt_stations_df["Opening date"].apply(lambda x: x.split(" ")[2])
mrt_stations_df["Opening year"] = mrt_stations_df["Opening year"].apply(lambda x: int(x[0:4]))
mrt_stations_df = mrt_stations_df.sort_values(by=["Opening year","Opening month"], ascending = [True,True])
mrt_stations_df = mrt_stations_df.drop_duplicates(subset="Name", keep="first")
mrt_stations_df = mrt_stations_df.reset_index(drop = True)
mrt_stations_df = mrt_stations_df[["Name","Opening year","Opening month"]]

print("Retrieving mrt station coordinates.....")
mrt_stations_coordinates = find_coordinates(mrt_stations_df)
mrt_stations_coordinates = mrt_stations_coordinates.rename(columns={"address":"Name"})
mrt_stations_coordinates = mrt_stations_coordinates[["Name","LATITUDE","LONGITUDE"]]
mrt_stations_coordinates = pd.merge(mrt_stations_coordinates, mrt_stations_df, how="left",on="Name")
mrt_stations_coordinates.to_csv("data/for_feature_engineering/mrt_stations/mrt_station_coordinates_w_period_test.csv",index=False)

###############
#### MALLS ####
###############
malls="https://en.wikipedia.org/wiki/List_of_shopping_malls_in_Singapore"
response = requests.get(malls)
print(f"Malls: {response.status_code}")
soup = BeautifulSoup(response.text, 'html.parser')

malls = []
for i in soup.find_all('li'):
    if i.string != None:
        malls.append(i.string)
    else:
        malls.append(i.text)
        
malls_df = pd.DataFrame(malls)
malls_df.columns = ["Name"]
malls_df = malls_df.drop_duplicates().reset_index(drop = True)
malls_df = malls_df.iloc[45:212,].reset_index(drop=True)
patt = r'\[(.*?)\]'
malls_df["Name"] = malls_df["Name"].apply(lambda x: x.replace(["[{}]".format(i) for i in re.findall(patt, x)][0],"") if bool(re.search(patt, x)) else x)
print("Retrieving mall coordinates.....")     
mall_coordinates = find_coordinates(malls_df)
mall_coordinates.to_csv("data/for_feature_engineering/malls/mall_coordinates_test.csv", index=False)

###############
#### PARKS ####
###############
parks="https://en.wikipedia.org/wiki/List_of_parks_in_Singapore"
response = requests.get(parks)
print(f"Parks: {response.status_code}")
soup = BeautifulSoup(response.text, 'html.parser')
parks_table = pd.read_html(str(parks))
parks_df = pd.DataFrame(parks_table[2])
parks_df = parks_df[["Name"]]

print("Retrieving park coordinates.....")
parks_coordinates = find_coordinates(parks_df)
parks_coordinates.to_csv("data/for_feature_engineering/parks/park_coordinates_test.csv", index=False)

