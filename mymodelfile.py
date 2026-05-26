import re
import warnings
warnings.filterwarnings("ignore")

from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from sklearn.ensemble import (ExtraTreesRegressor,
                               HistGradientBoostingRegressor)
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


# ═══════════════════════════════════════════════════════════════════════════ #
#  EMBEDDED PLAYER DATA                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

PLAYER_DATA = {
    "150870": {"team_name": "Chennai Super Kings",        "delivery_name": "RD Gaikwad"},
    "741229": {"team_name": "Chennai Super Kings",        "delivery_name": "MS Dhoni"},
    "170868": {"team_name": "Chennai Super Kings",        "delivery_name": "SV Samson"},
    "244953": {"team_name": "Chennai Super Kings",        "delivery_name": "D Brevis"},
    "509617": {"team_name": "Chennai Super Kings",        "delivery_name": "AS Mhatre"},
    "293331": {"team_name": "Chennai Super Kings",        "delivery_name": "Kartik Sharma"},
    "164517": {"team_name": "Chennai Super Kings",        "delivery_name": "SN Khan"},
    "710840": {"team_name": "Chennai Super Kings",        "delivery_name": "Urvil Patel"},
    "318207": {"team_name": "Chennai Super Kings",        "delivery_name": "A Kamboj"},
    "469620": {"team_name": "Chennai Super Kings",        "delivery_name": "J Overton"},
    "198519": {"team_name": "Chennai Super Kings",        "delivery_name": "R Ghosh"},
    "326163": {"team_name": "Chennai Super Kings",        "delivery_name": "P Veer"},
    "481987": {"team_name": "Chennai Super Kings",        "delivery_name": "MW Short"},
    "146805": {"team_name": "Chennai Super Kings",        "delivery_name": "Aman Khan"},
    "136284": {"team_name": "Chennai Super Kings",        "delivery_name": "ZJ Foulkes"},
    "266935": {"team_name": "Chennai Super Kings",        "delivery_name": "SS Dube"},
    "135658": {"team_name": "Chennai Super Kings",        "delivery_name": "K Ahmed"},
    "187933": {"team_name": "Chennai Super Kings",        "delivery_name": "Noor Ahmad"},
    "299592": {"team_name": "Chennai Super Kings",        "delivery_name": "M Choudhary"},
    "193764": {"team_name": "Chennai Super Kings",        "delivery_name": "NM Ellis"},
    "279666": {"team_name": "Chennai Super Kings",        "delivery_name": "S Gopal"},
    "137077": {"team_name": "Chennai Super Kings",        "delivery_name": "Gurjapneet Singh"},
    "724994": {"team_name": "Chennai Super Kings",        "delivery_name": "A Hosein"},
    "102039": {"team_name": "Chennai Super Kings",        "delivery_name": "MJ Henry"},
    "682966": {"team_name": "Chennai Super Kings",        "delivery_name": "RA Chahar"},
    "259802": {"team_name": "Delhi Capitals",             "delivery_name": "KL Rahul"},
    "571205": {"team_name": "Delhi Capitals",             "delivery_name": "KK Nair"},
    "246384": {"team_name": "Delhi Capitals",             "delivery_name": "DA Miller"},
    "151078": {"team_name": "Delhi Capitals",             "delivery_name": "Ben Duckett"},
    "467498": {"team_name": "Delhi Capitals",             "delivery_name": "Pathum Nissanka"},
    "204837": {"team_name": "Delhi Capitals",             "delivery_name": "Sahil Parakh"},
    "290977": {"team_name": "Delhi Capitals",             "delivery_name": "PP Shaw"},
    "126079": {"team_name": "Delhi Capitals",             "delivery_name": "Abishek Porel"},
    "318917": {"team_name": "Delhi Capitals",             "delivery_name": "T Stubbs"},
    "276596": {"team_name": "Delhi Capitals",             "delivery_name": "AR Patel"},
    "794027": {"team_name": "Delhi Capitals",             "delivery_name": "Sameer Rizvi"},
    "198596": {"team_name": "Delhi Capitals",             "delivery_name": "Ashutosh Sharma"},
    "965679": {"team_name": "Delhi Capitals",             "delivery_name": "V Nigam"},
    "335784": {"team_name": "Delhi Capitals",             "delivery_name": "Ajay Mandal"},
    "152436": {"team_name": "Delhi Capitals",             "delivery_name": "Tripurana Vijay"},
    "617819": {"team_name": "Delhi Capitals",             "delivery_name": "M Tiwari"},
    "275131": {"team_name": "Delhi Capitals",             "delivery_name": "Auqib Dar"},
    "270824": {"team_name": "Delhi Capitals",             "delivery_name": "N Rana"},
    "353816": {"team_name": "Delhi Capitals",             "delivery_name": "MA Starc"},
    "137950": {"team_name": "Delhi Capitals",             "delivery_name": "T Natarajan"},
    "196257": {"team_name": "Delhi Capitals",             "delivery_name": "Mukesh Kumar"},
    "163994": {"team_name": "Delhi Capitals",             "delivery_name": "PVD Chameera"},
    "130760": {"team_name": "Delhi Capitals",             "delivery_name": "L Ngidi"},
    "855443": {"team_name": "Delhi Capitals",             "delivery_name": "KA Jamieson"},
    "647601": {"team_name": "Delhi Capitals",             "delivery_name": "Kuldeep Yadav"},
    "128885": {"team_name": "Gujarat Titans",             "delivery_name": "Shubman Gill"},
    "216569": {"team_name": "Gujarat Titans",             "delivery_name": "JC Buttler"},
    "720748": {"team_name": "Gujarat Titans",             "delivery_name": "Kumar Kushagra"},
    "106102": {"team_name": "Gujarat Titans",             "delivery_name": "Anuj Rawat"},
    "175604": {"team_name": "Gujarat Titans",             "delivery_name": "T Banton"},
    "327222": {"team_name": "Gujarat Titans",             "delivery_name": "GD Phillips"},
    "126848": {"team_name": "Gujarat Titans",             "delivery_name": "Nishant Sindhu"},
    "104901": {"team_name": "Gujarat Titans",             "delivery_name": "Washington Sundar"},
    "169129": {"team_name": "Gujarat Titans",             "delivery_name": "Arshad Khan"},
    "149599": {"team_name": "Gujarat Titans",             "delivery_name": "R Sai Kishore"},
    "296158": {"team_name": "Gujarat Titans",             "delivery_name": "J Yadav"},
    "127132": {"team_name": "Gujarat Titans",             "delivery_name": "JO Holder"},
    "315909": {"team_name": "Gujarat Titans",             "delivery_name": "B Sai Sudharsan"},
    "145075": {"team_name": "Gujarat Titans",             "delivery_name": "M Shahrukh Khan"},
    "239140": {"team_name": "Gujarat Titans",             "delivery_name": "K Rabada"},
    "336139": {"team_name": "Gujarat Titans",             "delivery_name": "Mohammed Siraj"},
    "432230": {"team_name": "Gujarat Titans",             "delivery_name": "Prasidh Krishna"},
    "909076": {"team_name": "Gujarat Titans",             "delivery_name": "MJ Suthar"},
    "236144": {"team_name": "Gujarat Titans",             "delivery_name": "Gurnoor Brar"},
    "143555": {"team_name": "Gujarat Titans",             "delivery_name": "I Sharma"},
    "191440": {"team_name": "Gujarat Titans",             "delivery_name": "Ashok Sharma"},
    "304981": {"team_name": "Gujarat Titans",             "delivery_name": "Y Prithvi Raj"},
    "141525": {"team_name": "Gujarat Titans",             "delivery_name": "L Wood"},
    "116827": {"team_name": "Gujarat Titans",             "delivery_name": "R Tewatia"},
    "605937": {"team_name": "Gujarat Titans",             "delivery_name": "Rashid Khan"},
    "458396": {"team_name": "Kolkata Knight Riders",      "delivery_name": "AM Rahane"},
    "282804": {"team_name": "Kolkata Knight Riders",      "delivery_name": "RK Singh"},
    "366647": {"team_name": "Kolkata Knight Riders",      "delivery_name": "A Raghuvanshi"},
    "187824": {"team_name": "Kolkata Knight Riders",      "delivery_name": "MK Pandey"},
    "286119": {"team_name": "Kolkata Knight Riders",      "delivery_name": "C Green"},
    "109519": {"team_name": "Kolkata Knight Riders",      "delivery_name": "FA Allen"},
    "788236": {"team_name": "Kolkata Knight Riders",      "delivery_name": "Tejasvi Singh"},
    "294349": {"team_name": "Kolkata Knight Riders",      "delivery_name": "RA Tripathi"},
    "155498": {"team_name": "Kolkata Knight Riders",      "delivery_name": "TL Seifert"},
    "128684": {"team_name": "Kolkata Knight Riders",      "delivery_name": "R Powell"},
    "791181": {"team_name": "Kolkata Knight Riders",      "delivery_name": "AS Roy"},
    "214037": {"team_name": "Kolkata Knight Riders",      "delivery_name": "Sarthak Ranjan"},
    "108626": {"team_name": "Kolkata Knight Riders",      "delivery_name": "Daksh Kamra"},
    "294525": {"team_name": "Kolkata Knight Riders",      "delivery_name": "R Ravindra"},
    "812047": {"team_name": "Kolkata Knight Riders",      "delivery_name": "Ramandeep Singh"},
    "262763": {"team_name": "Kolkata Knight Riders",      "delivery_name": "Blessing Muzarabani"},
    "319082": {"team_name": "Kolkata Knight Riders",      "delivery_name": "VG Arora"},
    "245746": {"team_name": "Kolkata Knight Riders",      "delivery_name": "M Pathirana"},
    "959522": {"team_name": "Kolkata Knight Riders",      "delivery_name": "Kartik Tyagi"},
    "332728": {"team_name": "Kolkata Knight Riders",      "delivery_name": "PH Solanki"},
    "220206": {"team_name": "Kolkata Knight Riders",      "delivery_name": "Akash Deep"},
    "119348": {"team_name": "Kolkata Knight Riders",      "delivery_name": "Harshit Rana"},
    "266331": {"team_name": "Kolkata Knight Riders",      "delivery_name": "Umran Malik"},
    "101983": {"team_name": "Kolkata Knight Riders",      "delivery_name": "SP Narine"},
    "229583": {"team_name": "Kolkata Knight Riders",      "delivery_name": "CV Varun"},
    "183262": {"team_name": "Lucknow Super Giants",       "delivery_name": "RR Pant"},
    "202076": {"team_name": "Lucknow Super Giants",       "delivery_name": "AK Markram"},
    "209599": {"team_name": "Lucknow Super Giants",       "delivery_name": "Himmat Singh"},
    "126589": {"team_name": "Lucknow Super Giants",       "delivery_name": "MP Breetzke"},
    "321255": {"team_name": "Lucknow Super Giants",       "delivery_name": "Mukesh Choudhary"},
    "203422": {"team_name": "Lucknow Super Giants",       "delivery_name": "Akshat Raghuwanshi"},
    "193130": {"team_name": "Lucknow Super Giants",       "delivery_name": "JP Inglis"},
    "139152": {"team_name": "Lucknow Super Giants",       "delivery_name": "N Pooran"},
    "217660": {"team_name": "Lucknow Super Giants",       "delivery_name": "MR Marsh"},
    "641937": {"team_name": "Lucknow Super Giants",       "delivery_name": "Abdul Samad"},
    "161043": {"team_name": "Lucknow Super Giants",       "delivery_name": "Shahbaz Ahamad"},
    "679116": {"team_name": "Lucknow Super Giants",       "delivery_name": "AA Kulkarni"},
    "146834": {"team_name": "Lucknow Super Giants",       "delivery_name": "Wanindu Hasaranga"},
    "124811": {"team_name": "Lucknow Super Giants",       "delivery_name": "A Badoni"},
    "291343": {"team_name": "Lucknow Super Giants",       "delivery_name": "Mohammed Shami"},
    "164615": {"team_name": "Lucknow Super Giants",       "delivery_name": "Avesh Khan"},
    "260103": {"team_name": "Lucknow Super Giants",       "delivery_name": "M Siddharth"},
    "233556": {"team_name": "Lucknow Super Giants",       "delivery_name": "DS Rathi"},
    "975228": {"team_name": "Lucknow Super Giants",       "delivery_name": "Akash Singh"},
    "193359": {"team_name": "Lucknow Super Giants",       "delivery_name": "Prince Yadav"},
    "410061": {"team_name": "Lucknow Super Giants",       "delivery_name": "Arjun Tendulkar"},
    "164742": {"team_name": "Lucknow Super Giants",       "delivery_name": "A Nortje"},
    "272347": {"team_name": "Lucknow Super Giants",       "delivery_name": "Naman Tiwari"},
    "691311": {"team_name": "Lucknow Super Giants",       "delivery_name": "MP Yadav"},
    "138667": {"team_name": "Lucknow Super Giants",       "delivery_name": "Mohsin Khan"},
    "622673": {"team_name": "Mumbai Indians",             "delivery_name": "RG Sharma"},
    "916276": {"team_name": "Mumbai Indians",             "delivery_name": "SA Yadav"},
    "560164": {"team_name": "Mumbai Indians",             "delivery_name": "R Minz"},
    "313650": {"team_name": "Mumbai Indians",             "delivery_name": "SE Rutherford"},
    "333593": {"team_name": "Mumbai Indians",             "delivery_name": "RD Rickelton"},
    "983691": {"team_name": "Mumbai Indians",             "delivery_name": "Q de Kock"},
    "136305": {"team_name": "Mumbai Indians",             "delivery_name": "Danish Malewar"},
    "165203": {"team_name": "Mumbai Indians",             "delivery_name": "Tilak Varma"},
    "195720": {"team_name": "Mumbai Indians",             "delivery_name": "HH Pandya"},
    "308406": {"team_name": "Mumbai Indians",             "delivery_name": "Naman Dhir"},
    "273086": {"team_name": "Mumbai Indians",             "delivery_name": "MJ Santner"},
    "262723": {"team_name": "Mumbai Indians",             "delivery_name": "RA Bawa"},
    "855744": {"team_name": "Mumbai Indians",             "delivery_name": "Atharva Ankolekar"},
    "286729": {"team_name": "Mumbai Indians",             "delivery_name": "Mayank Rawat"},
    "280380": {"team_name": "Mumbai Indians",             "delivery_name": "C Bosch"},
    "433815": {"team_name": "Mumbai Indians",             "delivery_name": "WG Jacks"},
    "138841": {"team_name": "Mumbai Indians",             "delivery_name": "SN Thakur"},
    "601224": {"team_name": "Mumbai Indians",             "delivery_name": "TA Boult"},
    "229447": {"team_name": "Mumbai Indians",             "delivery_name": "M Markande"},
    "261844": {"team_name": "Mumbai Indians",             "delivery_name": "DL Chahar"},
    "617494": {"team_name": "Mumbai Indians",             "delivery_name": "Ashwani Kumar"},
    "124623": {"team_name": "Mumbai Indians",             "delivery_name": "Raghu Sharma"},
    "240663": {"team_name": "Mumbai Indians",             "delivery_name": "Mohammad Izhar"},
    "260145": {"team_name": "Mumbai Indians",             "delivery_name": "Allah Ghazanfar"},
    "591090": {"team_name": "Mumbai Indians",             "delivery_name": "JJ Bumrah"},
    "454399": {"team_name": "Punjab Kings",               "delivery_name": "SS Iyer"},
    "175481": {"team_name": "Punjab Kings",               "delivery_name": "N Wadhera"},
    "157099": {"team_name": "Punjab Kings",               "delivery_name": "Vishnu Vinod"},
    "944609": {"team_name": "Punjab Kings",               "delivery_name": "Harnoor Pannu"},
    "128424": {"team_name": "Punjab Kings",               "delivery_name": "Pyla Avinash"},
    "157844": {"team_name": "Punjab Kings",               "delivery_name": "P Simran Singh"},
    "212990": {"team_name": "Punjab Kings",               "delivery_name": "Shashank Singh"},
    "331330": {"team_name": "Punjab Kings",               "delivery_name": "MP Stoinis"},
    "200898": {"team_name": "Punjab Kings",               "delivery_name": "Harpreet Brar"},
    "317613": {"team_name": "Punjab Kings",               "delivery_name": "M Jansen"},
    "332725": {"team_name": "Punjab Kings",               "delivery_name": "Azmatullah Omarzai"},
    "206083": {"team_name": "Punjab Kings",               "delivery_name": "Priyansh Arya"},
    "318346": {"team_name": "Punjab Kings",               "delivery_name": "Musheer Khan"},
    "270993": {"team_name": "Punjab Kings",               "delivery_name": "Suryansh Shedge"},
    "557197": {"team_name": "Punjab Kings",               "delivery_name": "MJ Owen"},
    "772759": {"team_name": "Punjab Kings",               "delivery_name": "Cooper Connolly"},
    "957219": {"team_name": "Punjab Kings",               "delivery_name": "Ben Dwarshuis"},
    "257209": {"team_name": "Punjab Kings",               "delivery_name": "Arshdeep Singh"},
    "110384": {"team_name": "Punjab Kings",               "delivery_name": "YS Chahal"},
    "117257": {"team_name": "Punjab Kings",               "delivery_name": "Vijaykumar Vyshak"},
    "920543": {"team_name": "Punjab Kings",               "delivery_name": "Yash Thakur"},
    "174024": {"team_name": "Punjab Kings",               "delivery_name": "XC Bartlett"},
    "114526": {"team_name": "Punjab Kings",               "delivery_name": "P Dubey"},
    "375016": {"team_name": "Punjab Kings",               "delivery_name": "Vishal Nishad"},
    "129274": {"team_name": "Punjab Kings",               "delivery_name": "LH Ferguson"},
    "104251": {"team_name": "Rajasthan Royals",           "delivery_name": "R Parag"},
    "470254": {"team_name": "Rajasthan Royals",           "delivery_name": "SB Dubey"},
    "139776": {"team_name": "Rajasthan Royals",           "delivery_name": "V Suryavanshi"},
    "450112": {"team_name": "Rajasthan Royals",           "delivery_name": "D Ferreira"},
    "975890": {"team_name": "Rajasthan Royals",           "delivery_name": "Lhuan-dre Pretorius"},
    "237260": {"team_name": "Rajasthan Royals",           "delivery_name": "Ravi Singh"},
    "111232": {"team_name": "Rajasthan Royals",           "delivery_name": "Aman Rao Perala"},
    "176306": {"team_name": "Rajasthan Royals",           "delivery_name": "SO Hetmyer"},
    "320949": {"team_name": "Rajasthan Royals",           "delivery_name": "YBK Jaiswal"},
    "319902": {"team_name": "Rajasthan Royals",           "delivery_name": "Dhruv Jurel"},
    "773524": {"team_name": "Rajasthan Royals",           "delivery_name": "Yudhvir Singh"},
    "259382": {"team_name": "Rajasthan Royals",           "delivery_name": "RA Jadeja"},
    "210755": {"team_name": "Rajasthan Royals",           "delivery_name": "SM Curran"},
    "550322": {"team_name": "Rajasthan Royals",           "delivery_name": "JC Archer"},
    "246413": {"team_name": "Rajasthan Royals",           "delivery_name": "TU Deshpande"},
    "642126": {"team_name": "Rajasthan Royals",           "delivery_name": "KT Maphaka"},
    "217849": {"team_name": "Rajasthan Royals",           "delivery_name": "Ravi Bishnoi"},
    "281054": {"team_name": "Rajasthan Royals",           "delivery_name": "Sushant Mishra"},
    "444298": {"team_name": "Rajasthan Royals",           "delivery_name": "Yash Raj Punja"},
    "147302": {"team_name": "Rajasthan Royals",           "delivery_name": "V Puthur"},
    "213375": {"team_name": "Rajasthan Royals",           "delivery_name": "Brijesh Sharma"},
    "315684": {"team_name": "Rajasthan Royals",           "delivery_name": "AF Milne"},
    "825252": {"team_name": "Rajasthan Royals",           "delivery_name": "KR Sen"},
    "444576": {"team_name": "Rajasthan Royals",           "delivery_name": "Sandeep Sharma"},
    "339246": {"team_name": "Rajasthan Royals",           "delivery_name": "N Burger"},
    "311358": {"team_name": "Royal Challengers Bengaluru","delivery_name": "RM Patidar"},
    "244029": {"team_name": "Royal Challengers Bengaluru","delivery_name": "D Padikkal"},
    "762818": {"team_name": "Royal Challengers Bengaluru","delivery_name": "V Kohli"},
    "110855": {"team_name": "Royal Challengers Bengaluru","delivery_name": "PD Salt"},
    "221800": {"team_name": "Royal Challengers Bengaluru","delivery_name": "JM Sharma"},
    "236091": {"team_name": "Royal Challengers Bengaluru","delivery_name": "Jordan Cox"},
    "154627": {"team_name": "Royal Challengers Bengaluru","delivery_name": "KH Pandya"},
    "670921": {"team_name": "Royal Challengers Bengaluru","delivery_name": "Swapnil Singh"},
    "759873": {"team_name": "Royal Challengers Bengaluru","delivery_name": "TH David"},
    "132914": {"team_name": "Royal Challengers Bengaluru","delivery_name": "R Shepherd"},
    "292157": {"team_name": "Royal Challengers Bengaluru","delivery_name": "JG Bethell"},
    "306734": {"team_name": "Royal Challengers Bengaluru","delivery_name": "VR Iyer"},
    "780887": {"team_name": "Royal Challengers Bengaluru","delivery_name": "Satvik Deswal"},
    "203920": {"team_name": "Royal Challengers Bengaluru","delivery_name": "Mangesh Yadav"},
    "267531": {"team_name": "Royal Challengers Bengaluru","delivery_name": "Vicky Ostwal"},
    "166660": {"team_name": "Royal Challengers Bengaluru","delivery_name": "Vihaan Malhotra"},
    "296935": {"team_name": "Royal Challengers Bengaluru","delivery_name": "Kanishk Chouhan"},
    "239005": {"team_name": "Royal Challengers Bengaluru","delivery_name": "JR Hazlewood"},
    "147190": {"team_name": "Royal Challengers Bengaluru","delivery_name": "Rasikh Salam"},
    "263028": {"team_name": "Royal Challengers Bengaluru","delivery_name": "Suyash Sharma"},
    "206736": {"team_name": "Royal Challengers Bengaluru","delivery_name": "B Kumar"},
    "580004": {"team_name": "Royal Challengers Bengaluru","delivery_name": "N Thushara"},
    "507034": {"team_name": "Royal Challengers Bengaluru","delivery_name": "Abhinandan Singh"},
    "171062": {"team_name": "Royal Challengers Bengaluru","delivery_name": "Jacob Duffy"},
    "590249": {"team_name": "Royal Challengers Bengaluru","delivery_name": "Yash Dayal"},
    "329687": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Ishan Kishan"},
    "114838": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Aniket Verma"},
    "114479": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Smaran Ravichandran"},
    "194311": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Salil Arora"},
    "213139": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "H Klaasen"},
    "161636": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "TM Head"},
    "169291": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "HV Patel"},
    "338673": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "PHKD Mendis"},
    "250925": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Harsh Dubey"},
    "838954": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Brydon Carse"},
    "337049": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Shivang Kumar"},
    "832130": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Krains Fuletra"},
    "276289": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "LS Livingstone"},
    "965651": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Jack Edwards"},
    "577813": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Abhishek Sharma"},
    "299161": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Nithish Kumar Reddy"},
    "212954": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "PJ Cummins"},
    "124291": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Zeeshan Ansari"},
    "248044": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "JD Unadkat"},
    "242331": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "E Malinga"},
    "326386": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Sakib Hussain"},
    "115758": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Onkar Tarmale"},
    "187064": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Amit Kumar"},
    "108444": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Praful Hinge"},
    "270556": {"team_name": "Sunrisers Hyderabad",        "delivery_name": "Shivam Mavi"},
}

_ID_TO_NAME = {pid: v["delivery_name"] for pid, v in PLAYER_DATA.items()}

# ─── Home grounds ─────────────────────────────────────────────────────────── #
_HOME_GROUNDS = {
    "Chennai Super Kings":         "chepauk",
    "Royal Challengers Bengaluru": "chinnaswamy",
    "Royal Challengers Bangalore": "chinnaswamy",
    "Mumbai Indians":              "wankhede",
    "Kolkata Knight Riders":       "eden",
    "Delhi Capitals":              "kotla",
    "Delhi Daredevils":            "kotla",
    "Rajasthan Royals":            "sawai",
    "Punjab Kings":                "mullanpur",   # ← updated: new home since 2024
    "Kings XI Punjab":             "bindra",
    "Sunrisers Hyderabad":         "uppal",
    "Gujarat Titans":              "narendra",
    "Lucknow Super Giants":        "ekana",
}

# ─── Venue nature ─────────────────────────────────────────────────────────── #
_VENUE_NATURE = {
    "chinnaswamy": "aggressive batting",
    "wankhede":    "batting + dew",
    "chepauk":     "spin-balanced",
    "eden":        "batting",
    "narendra":    "batting",
    "kotla":       "extreme batting",
    "uppal":       "ultra batting",
    "ekana":       "slow-balanced",
    "sawai":       "balanced",
    "barsapara":   "batting",
  
    "himachal":    "pace friendly",  
    "mullanpur":   "balanced",        
    "bindra":      "balanced",      
}   

_NATURE_SCORE = {
    "aggressive batting": 1.0,
    "extreme batting":    1.2,
    "ultra batting":      1.4,
    "batting + dew":      0.8,
    "batting":            0.7,
    "balanced":           0.0,
    "pace friendly":     -0.6,      
    "spin-balanced":     -0.3,
    "spin/bowling":      -0.6,
    "slow-balanced":     -0.4,
    "slow/bowling":      -0.7,
}

_VENUE_HOME_OVERRIDE = {
    "chepauk":    (0.90, -0.15),
    "ekana":      (0.95, -0.20),
    "bindra":     (1.00, -0.10),
    "mullanpur":  (1.00, -0.10),
}

_AWAY_DISCOUNT = {
    "chinnaswamy": 0.39, "uppal": 0.30, "kotla": 0.28,
    "eden": 0.18, "wankhede": 0.18, "narendra": 0.12,
    "barsapara": 0.18, "himachal": 0.08,  # Dharamsala: modest advantage for pace teams visiting
    "sawai": 0.15,
    "chepauk": 0.05, "ekana": 0.05, "bindra": 0.10,
    "mullanpur": 0.10,
}

_TEAM_PP_AGGRESSION = {
    "Sunrisers Hyderabad":         1.40,   # ↑ Head + Abhishek historically fastest starters
    "Royal Challengers Bengaluru": 1.28,   # ↑ Salt + Kohli/Patidar, 2026 season form
    "Delhi Capitals":              1.18,
    "Kolkata Knight Riders":       1.08,
    "Punjab Kings":                1.12,
    "Rajasthan Royals":            1.14,   # ↑ Jaiswal + Suryavanshi very aggressive
    "Mumbai Indians":              1.02,
    "Lucknow Super Giants":        1.00,
    "Gujarat Titans":              0.96,   # ↓ GT more measured; Gill/Sudharsan methodical
    "Chennai Super Kings":         0.97,
}

_TEAM_PP_BOWLING_QUALITY = {
    "Mumbai Indians":              0.80,
    "Royal Challengers Bengaluru": 0.82,   # ↑ Hazlewood + Bhuvi among best PP attacks in 2026
    "Gujarat Titans":              0.84,   # ↑ Rabada + Siraj very effective in PP
    "Kolkata Knight Riders":       0.90,
    "Rajasthan Royals":            0.88,   # ↑ Archer/Sandeep/Maphaka tight in PP
    "Delhi Capitals":              0.94,
    "Sunrisers Hyderabad":         0.93,
    "Punjab Kings":                0.95,
    "Lucknow Super Giants":        1.00,
    "Chennai Super Kings":         1.02,
}

# ─── Team batting correction (updated for IPL 2026 season) ──────────────── #
# Based on each team's 2026 league stage PP performance vs baseline
_TEAM_BATTING_CORRECTION_2026 = {
    "Royal Challengers Bengaluru": +12.5,  # ↑↑ dominant season, finished 1st
    "Gujarat Titans":              +8.0,   # ↑ solid; Gill/Buttler/Sudharsan
    "Sunrisers Hyderabad":         +6.5,   # consistent but not as explosive as 2024
    "Rajasthan Royals":            +5.0,   # Jaiswal + Parag in form
    "Lucknow Super Giants":        +14.0,
    "Chennai Super Kings":         +9.5,
    "Punjab Kings":                +5.5,
    "Mumbai Indians":              +4.0,
    "Kolkata Knight Riders":       +2.0,
    "Delhi Capitals":              +2.0,
}

_INN2_CHASE_AGGRESSION_BASE = 6.0
_INN2_CHASE_HIGH_TARGET_THRESH = 65.0
_INN2_CHASE_HIGH_TARGET_BONUS  = 5.0

# ─── Matchup calibration ──────────────────────────────────────────────────── #
# Added Dharamsala (himachal) and Mullanpur (mullanpur) matchups for 2026 playoffs
_MATCHUP_CALIBRATION = {
    # ── Existing venue matchups ─────────────────────────────────────────── #
    ("Mumbai Indians", "Chennai Super Kings", "wankhede"):              {"inn1_adj": +5.5,  "inn2_adj": +3.5,  "confidence": 0.65},
    ("Chennai Super Kings", "Mumbai Indians", "wankhede"):              {"inn1_adj": -4.0,  "inn2_adj": -2.5,  "confidence": 0.60},
    ("Royal Challengers Bengaluru", "Gujarat Titans", "chinnaswamy"):   {"inn1_adj": +8.0,  "inn2_adj": +5.5,  "confidence": 0.72},
    ("Gujarat Titans", "Royal Challengers Bengaluru", "chinnaswamy"):   {"inn1_adj": -7.5,  "inn2_adj": -4.0,  "confidence": 0.70},
    ("Delhi Capitals", "Punjab Kings", "kotla"):                        {"inn1_adj": +6.0,  "inn2_adj": +4.0,  "confidence": 0.68},
    ("Punjab Kings", "Delhi Capitals", "kotla"):                        {"inn1_adj": +3.5,  "inn2_adj": +2.5,  "confidence": 0.55},
    ("Rajasthan Royals", "Sunrisers Hyderabad", "sawai"):               {"inn1_adj": +4.0,  "inn2_adj": +2.5,  "confidence": 0.62},
    ("Sunrisers Hyderabad", "Rajasthan Royals", "sawai"):               {"inn1_adj": +6.5,  "inn2_adj": +4.0,  "confidence": 0.68},
    ("Chennai Super Kings", "Gujarat Titans", "chinnaswamy"):           {"inn1_adj": -5.0,  "inn2_adj": -3.0,  "confidence": 0.60},
    ("Gujarat Titans", "Chennai Super Kings", "chinnaswamy"):           {"inn1_adj": -4.0,  "inn2_adj": -2.5,  "confidence": 0.58},
    ("Lucknow Super Giants", "Kolkata Knight Riders", "eden"):          {"inn1_adj": -3.0,  "inn2_adj": -1.5,  "confidence": 0.58},
    ("Kolkata Knight Riders", "Lucknow Super Giants", "eden"):          {"inn1_adj": +4.5,  "inn2_adj": +3.0,  "confidence": 0.65},
    ("Delhi Capitals", "Royal Challengers Bengaluru", "kotla"):         {"inn1_adj": +7.0,  "inn2_adj": +5.0,  "confidence": 0.70},
    ("Royal Challengers Bengaluru", "Delhi Capitals", "kotla"):         {"inn1_adj": +5.5,  "inn2_adj": +4.0,  "confidence": 0.65},
    ("Sunrisers Hyderabad", "Mumbai Indians", "uppal"):                 {"inn1_adj": +8.5,  "inn2_adj": +5.0,  "confidence": 0.70},
    ("Mumbai Indians", "Sunrisers Hyderabad", "uppal"):                 {"inn1_adj": -6.0,  "inn2_adj": -3.5,  "confidence": 0.65},
    ("Royal Challengers Bengaluru", "Mumbai Indians", "chinnaswamy"):   {"inn1_adj": +7.5,  "inn2_adj": +5.0,  "confidence": 0.70},
    ("Kolkata Knight Riders", "Mumbai Indians", "eden"):                {"inn1_adj": +5.0,  "inn2_adj": +3.5,  "confidence": 0.63},
    ("Punjab Kings", "Mumbai Indians", "bindra"):                       {"inn1_adj": +2.0,  "inn2_adj": +1.5,  "confidence": 0.50},
    ("Rajasthan Royals", "Mumbai Indians", "sawai"):                    {"inn1_adj": +3.5,  "inn2_adj": +2.0,  "confidence": 0.58},
    ("Chennai Super Kings", "Mumbai Indians", "chepauk"):               {"inn1_adj": +30.0, "inn2_adj": +12.0, "confidence": 0.80},
    ("Mumbai Indians", "Chennai Super Kings", "chepauk"):               {"inn1_adj": -3.0,  "inn2_adj": -2.0,  "confidence": 0.60},
    ("Punjab Kings", "Delhi Capitals", "kotla"):                        {"inn1_adj": +12.0, "inn2_adj": +8.0,  "confidence": 0.70},
    ("Sunrisers Hyderabad", "Rajasthan Royals", "uppal"):               {"inn1_adj": +10.0, "inn2_adj": +7.0,  "confidence": 0.72},


    ("Royal Challengers Bengaluru", "Gujarat Titans", "himachal"):      {"inn1_adj": +5.0,  "inn2_adj": +3.5,  "confidence": 0.62},
    ("Gujarat Titans", "Royal Challengers Bengaluru", "himachal"):      {"inn1_adj": -4.5,  "inn2_adj": -2.5,  "confidence": 0.60},

   
    ("Sunrisers Hyderabad", "Rajasthan Royals", "mullanpur"):           {"inn1_adj": +4.0,  "inn2_adj": +2.5,  "confidence": 0.58},
    ("Rajasthan Royals", "Sunrisers Hyderabad", "mullanpur"):           {"inn1_adj": +2.5,  "inn2_adj": +4.0,  "confidence": 0.60}, 

   
    ("Punjab Kings", "Rajasthan Royals", "mullanpur"):                  {"inn1_adj": +3.5,  "inn2_adj": +2.0,  "confidence": 0.55},
    ("Rajasthan Royals", "Punjab Kings", "mullanpur"):                  {"inn1_adj": +1.5,  "inn2_adj": +2.5,  "confidence": 0.52},
    ("Punjab Kings", "Sunrisers Hyderabad", "mullanpur"):               {"inn1_adj": +2.0,  "inn2_adj": +1.5,  "confidence": 0.50},
}

# ─── High / low scoring venue prior (added mullanpur + himachal) ─────────── #
_HIGH_SCORING_VENUES = {
    "kotla": 85.0, "uppal": 87.5, "chinnaswamy": 80.0,
    "wankhede": 75.0, "narendra": 80.0, "eden": 75.0,
    "barsapara": 80.0, "sawai": 72.0,
  
    "himachal": 67.0,    
    "mullanpur": 68.5,   
}
_LOW_SCORING_VENUES = {"ekana": 52.0, "bindra": 62.0}

_GLOBAL_2026_PP_OFFSET = 6.0  

_HOME_PP_CALIBRATION = {
    ("Chennai Super Kings",         "chepauk"):      {"inn1_boost": 30.0, "inn2_boost": 12.0, "conf": 0.80},
    ("Royal Challengers Bengaluru", "chinnaswamy"):  {"inn1_boost":  7.0, "inn2_boost":  4.0, "conf": 0.52},
    ("Sunrisers Hyderabad",         "uppal"):        {"inn1_boost": 10.0, "inn2_boost":  6.0, "conf": 0.60},
    ("Delhi Capitals",              "kotla"):        {"inn1_boost":  6.0, "inn2_boost":  3.5, "conf": 0.50},
    ("Kolkata Knight Riders",       "eden"):         {"inn1_boost":  4.0, "inn2_boost":  2.5, "conf": 0.48},
    ("Mumbai Indians",              "wankhede"):     {"inn1_boost":  4.0, "inn2_boost":  2.5, "conf": 0.48},
    ("Rajasthan Royals",            "sawai"):        {"inn1_boost":  3.5, "inn2_boost":  2.0, "conf": 0.45},
    ("Lucknow Super Giants",        "ekana"):        {"inn1_boost": -2.0, "inn2_boost": -1.5, "conf": 0.45},
    # Punjab Kings now at Mullanpur
    ("Punjab Kings",                "mullanpur"):    {"inn1_boost":  3.0, "inn2_boost":  2.0, "conf": 0.45},
    ("Punjab Kings",                "bindra"):       {"inn1_boost":  1.5, "inn2_boost":  1.0, "conf": 0.35},  # legacy
    ("Gujarat Titans",              "narendra"):     {"inn1_boost":  3.5, "inn2_boost":  2.0, "conf": 0.45},
    # No home advantage at Dharamsala or Mullanpur for playoff teams (RCB, GT, SRH, RR)
}

_SPIN_SUPPRESSION_VENUES = {"chepauk", "ekana", "eden"}


_PLAYOFF_PRESSURE_DISCOUNT = -4.0

_DEW_INN2_BONUS = {
    "himachal":  4.5,
    "mullanpur": 5.0,  
    "wankhede":  5.0,
    "narendra":  3.5,
    "uppal":     4.0,
    "kotla":     3.5,
    "chinnaswamy": 3.0,
    "eden":      3.0,
}

VALID_VENUES = [
    'M Chinnaswamy Stadium', 'Punjab Cricket Association Stadium, Mohali',
    'Feroz Shah Kotla', 'Eden Gardens', 'Wankhede Stadium',
    'Sawai Mansingh Stadium', 'Rajiv Gandhi International Stadium, Uppal',
    'MA Chidambaram Stadium, Chepauk', 'Dr DY Patil Sports Academy',
    'Newlands', "St George\u2019s Park", 'Kingsmead', 'SuperSport Park',
    'Buffalo Park', 'New Wanderers Stadium', 'De Beers Diamond Oval',
    'OUTsurance Oval', 'Brabourne Stadium', 'Sardar Patel Stadium, Motera',
    'Barabati Stadium', 'Brabourne Stadium, Mumbai',
    'Vidarbha Cricket Association Stadium, Jamtha',
    'Himachal Pradesh Cricket Association Stadium', 'Nehru Stadium',
    'Holkar Cricket Stadium',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
    'Subrata Roy Sahara Stadium', 'Maharashtra Cricket Association Stadium',
    'Shaheed Veer Narayan Singh International Stadium',
    'JSCA International Stadium Complex', 'Sheikh Zayed Stadium',
    'Sharjah Cricket Stadium', 'Dubai International Cricket Stadium',
    'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'Saurashtra Cricket Association Stadium', 'Green Park',
    'M.Chinnaswamy Stadium', 'Punjab Cricket Association IS Bindra Stadium',
    'Rajiv Gandhi International Stadium', 'MA Chidambaram Stadium',
    'Arun Jaitley Stadium', 'MA Chidambaram Stadium, Chepauk, Chennai',
    'Wankhede Stadium, Mumbai', 'Narendra Modi Stadium, Ahmedabad',
    'Arun Jaitley Stadium, Delhi', 'Zayed Cricket Stadium, Abu Dhabi',
    'Dr DY Patil Sports Academy, Mumbai',
    'Maharashtra Cricket Association Stadium, Pune', 'Eden Gardens, Kolkata',
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh',
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad',
    'M Chinnaswamy Stadium, Bengaluru', 'Barsapara Cricket Stadium, Guwahati',
    'Sawai Mansingh Stadium, Jaipur',
    'Himachal Pradesh Cricket Association Stadium, Dharamsala',
    'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam',
    # ↓ Additional aliases for 2026 playoff venues
    'HPCA Stadium, Dharamsala',
    'Himachal Pradesh Cricket Association Stadium, Dharamsala, HP',
    'Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh',
    'New PCA Stadium, Mullanpur',
    'PCA Stadium, Mullanpur, New Chandigarh',
]

DELIVERIES_CSV_PATH = "/var/deliveries_updated_ipl_upto_2025.csv"
MATCHES_CSV_PATH    = "/var/matches_updated_ipl_upto_2025.csv"
MATCHES_USECOLS     = ["matchId", "venue", "city", "date", "season",
                        "toss_winner", "toss_decision"]

_STADIUM_PP_DATA: dict     = {}
_STADIUM_VENUE_INDEX: dict = {}


# ═══════════════════════════════════════════════════════════════════════════ #
#  UTILITIES                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

def _normalize_venue(v: str) -> str:
    v = v.lower()
    v = re.sub(r"[,\.'\u2019\-]", " ", v)
    return re.sub(r"\s+", " ", v).strip()


def _build_venue_index(known_venues):
    return {_normalize_venue(v): v for v in known_venues}


def _fuzzy_match_venue(query, venue_index, threshold=0.55):
    q = _normalize_venue(query)
    if q in venue_index:
        return venue_index[q]
    for nk, orig in venue_index.items():
        if q in nk or nk in q:
            return orig
    STOP = {"stadium", "ground", "cricket", "international", "the", "of", "and", "at"}
    _UNIQUE = {
        "jsca", "ekana", "chepauk", "chinnaswamy", "kotla", "wankhede",
        "brabourne", "barabati", "kingsmead", "newlands", "motera",
        "mullanpur", "dharamsala", "guwahati", "visakhapatnam",
        "mohali", "sharjah", "dubai", "uppal", "jamtha", "zayed",
        "barsapara", "vidarbha", "subrata", "shaheed", "himachal",
        "hpca", "maharaja", "yadavindra",
    }
    qt = set(q.split()) - STOP
    best_o, best_s = None, 0.0
    for nk, orig in venue_index.items():
        kt = set(nk.split()) - STOP
        sh = qt & kt
        n  = len(sh)
        if n >= 2:
            sc = n / max(len(qt), len(kt), 1)
            if sc > best_s:
                best_s, best_o = sc, orig
        elif n == 1:
            tok = next(iter(sh))
            if tok in _UNIQUE and 0.5 > best_s:
                best_s, best_o = 0.5, orig
    if best_o:
        return best_o
    best_o, best_s = None, 0.0
    for nk, orig in venue_index.items():
        r = SequenceMatcher(None, q, nk).ratio()
        if r > best_s:
            best_s, best_o = r, orig
    return best_o if best_s >= threshold else None


_VALID_VENUE_INDEX = {_normalize_venue(v): v for v in VALID_VENUES}


def validate_venue(raw_venue, threshold=0.72):
    m = _fuzzy_match_venue(raw_venue, _VALID_VENUE_INDEX, threshold)
    return m if m is not None else raw_venue


def _is_home_ground(venue: str, bat_team: str) -> bool:
    key = _HOME_GROUNDS.get(bat_team, "")
    return bool(key) and key in _normalize_venue(venue)


def _get_venue_key(venue: str):
    nv = _normalize_venue(venue)
    all_venue_keys = (
        list(_HIGH_SCORING_VENUES.keys())
        + list(_LOW_SCORING_VENUES.keys())
        + ["himachal", "mullanpur", "bindra", "uppal", "chepauk",
           "chinnaswamy", "wankhede", "eden", "narendra", "kotla",
           "sawai", "ekana", "barsapara"]
    )
    for key in all_venue_keys:
        if key in nv:
            return key
    return None


def _get_venue_prior(venue: str) -> float:
    sd = _get_stadium_data(venue)
    if sd is not None:
        return sd["inn1_avg"]
    key = _get_venue_key(venue)
    if key in _HIGH_SCORING_VENUES:
        return _HIGH_SCORING_VENUES[key]
    if key in _LOW_SCORING_VENUES:
        return _LOW_SCORING_VENUES[key]
    return 55.0


def _get_stadium_data(venue: str):
    if not _STADIUM_PP_DATA or not _STADIUM_VENUE_INDEX:
        return None
    m = _fuzzy_match_venue(venue, _STADIUM_VENUE_INDEX, threshold=0.50)
    return _STADIUM_PP_DATA.get(m) if m else None


def _recency_weight(year) -> float:
    if year >= 2026: return 25.0   
    if year >= 2025: return 20.0
    if year >= 2024: return 8.0
    if year >= 2023: return 5.0
    if year >= 2022: return 3.0
    if year >= 2020: return 1.5
    if year >= 2018: return 0.8
    return 0.4


def _wmean(values, weights) -> float:
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values,  dtype=float)
    s = w.sum()
    return float(np.dot(v, w) / s) if s > 0 else (float(v.mean()) if len(v) else 0.0)


def _shrink(obs, prior, n, k=10) -> float:
    k_eff = k / (1.0 + 0.1 * np.log1p(n))
    return float((n / (n + k_eff)) * obs + (k_eff / (n + k_eff)) * prior)


def _trend_slope(pairs) -> float:
    if len(pairs) < 3:
        return 0.0
    pairs = sorted(pairs)
    s = np.array([p[0] for p in pairs])
    v = np.array([p[1] for p in pairs])
    w = np.array([_recency_weight(x) for x in s])
    w /= w.sum()
    coef = np.polyfit(s - s.mean(), v, 1, w=w)
    return float(np.clip(coef[0], -4, 4))


def _get_matchup_calibration(bat_team, bowl_team, venue):
    venue_key = _get_venue_key(venue) or ""
    for (bf, bwf, vk), cal in _MATCHUP_CALIBRATION.items():
        if (vk in venue_key
                and bf.lower() in bat_team.lower()
                and bwf.lower() in bowl_team.lower()):
            return cal["inn1_adj"], cal["inn2_adj"], cal["confidence"]
    return 0.0, 0.0, 0.0


def _get_team_pp_aggression(team_name: str) -> float:
    for k, v in _TEAM_PP_AGGRESSION.items():
        if k.lower() in team_name.lower() or team_name.lower() in k.lower():
            return v
    return 1.0


def _get_team_pp_bowling_quality(team_name: str) -> float:
    for k, v in _TEAM_PP_BOWLING_QUALITY.items():
        if k.lower() in team_name.lower() or team_name.lower() in k.lower():
            return v
    return 1.0


def _derive_nature(inn1_avg: float) -> str:
    if inn1_avg >= 85: return "ultra batting"
    if inn1_avg >= 78: return "aggressive batting"
    if inn1_avg >= 72: return "batting"
    if inn1_avg >= 65: return "balanced"
    if inn1_avg >= 55: return "pace friendly"
    return "spin-balanced"


def _away_team_pp_profile(bat_team: str, venue: str) -> float:
    if _is_home_ground(venue, bat_team):
        return 0.0
    venue_prior = _get_venue_prior(venue)
    venue_key   = _get_venue_key(venue)
    if venue_key and venue_key in _AWAY_DISCOUNT:
        away_adj = -(_AWAY_DISCOUNT[venue_key] * venue_prior)
    else:
        excess   = venue_prior - 55.0
        disc     = min(0.010 * excess, 0.25) if excess >= 0 else 0.0
        away_adj = -(disc * venue_prior) if excess >= 0 else 0.10 * excess
    aggression = _get_team_pp_aggression(bat_team)
    away_adj   = away_adj * (2.0 - aggression)
    return float(np.clip(away_adj, -30, 0))


def _exp_rolling(values, half_life=3.0) -> float:
    if len(values) == 0:
        return 0.0
    n   = len(values)
    idx = np.arange(n)
    w   = 2.0 ** ((idx - (n - 1)) / half_life)
    return float(np.dot(w, values) / w.sum())


def _effective_nature_score(venue_key: str, nature: str, bat_team: str) -> float:
    base_score = _NATURE_SCORE.get(nature, 0.0)
    if not _is_home_ground(venue_key if venue_key else "", bat_team):
        return base_score
    override = _VENUE_HOME_OVERRIDE.get(venue_key)
    if override is None:
        return base_score
    agg_thresh, max_penalty = override
    team_agg = _get_team_pp_aggression(bat_team)
    if team_agg >= agg_thresh and base_score < max_penalty:
        return max_penalty
    return base_score


def _get_home_pp_calibration(bat_team: str, venue_key: str):
    for (team_frag, vk), cal in _HOME_PP_CALIBRATION.items():
        if (vk == venue_key
                and (team_frag.lower() in bat_team.lower()
                     or bat_team.lower() in team_frag.lower())):
            return cal["inn1_boost"], cal["inn2_boost"], cal["conf"]
    return 0.0, 0.0, 0.0


def _get_team_batting_correction(team_name: str) -> float:
    for k, v in _TEAM_BATTING_CORRECTION_2026.items():
        if k.lower() in team_name.lower() or team_name.lower() in k.lower():
            return v
    return 0.0


def _get_inn2_chase_boost(inn1_score: float) -> float:
    boost = _INN2_CHASE_AGGRESSION_BASE
    if inn1_score >= _INN2_CHASE_HIGH_TARGET_THRESH:
        boost += _INN2_CHASE_HIGH_TARGET_BONUS
    return boost


def _get_dew_inn2_bonus(venue: str) -> float:
    """Dew factor bonus for 2nd innings at night matches (relevant for all playoff venues)."""
    vk = _get_venue_key(venue)
    if vk:
        return _DEW_INN2_BONUS.get(vk, 2.5)
    return 2.5


# ═══════════════════════════════════════════════════════════════════════════ #
#  STADIUM PP DATA                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def _compute_stadium_pp_data(deliveries_df, matches_df,
                              primary_year: int = 2025,
                              lookback_years: int = 3):
    global _STADIUM_PP_DATA, _STADIUM_VENUE_INDEX
    MIN_MATCHES = 3

    d = deliveries_df.copy()
    d["batsman_runs"] = pd.to_numeric(d["batsman_runs"], errors="coerce").fillna(0)
    d["extras"]       = pd.to_numeric(d["extras"],       errors="coerce").fillna(0)
    d["isWide"]       = pd.to_numeric(d.get("isWide",  0), errors="coerce").fillna(0)
    d["isNoBall"]     = pd.to_numeric(d.get("isNoBall",0), errors="coerce").fillna(0)
    d["ball_runs"]    = d["batsman_runs"] + d["extras"]
    d["over"]         = pd.to_numeric(d.get("over", 0), errors="coerce").fillna(0).astype(int)
    d["year"]         = pd.to_datetime(d.get("date"), errors="coerce").dt.year.fillna(2020).astype(int)
    d["is_dismissal"] = 0
    if "player_dismissed" in d.columns:
        d["is_dismissal"] = d["player_dismissed"].apply(
            lambda x: 0 if (pd.isna(x) or str(x).strip() in ("", "nan")) else 1)

    if matches_df is not None and "venue" in matches_df.columns:
        vm = (matches_df[["matchId", "venue"]].drop_duplicates("matchId")
              .astype({"matchId": int, "venue": str}))
        d = d.merge(vm, on="matchId", how="left")

    if "venue" not in d.columns:
        d["venue"] = "Unknown"
    else:
        d["venue"] = d["venue"].fillna("Unknown")

    pp  = d[d["over"] < 6].copy()
    inn = (pp.groupby(["matchId", "inning", "venue", "year"], observed=True)
             .agg(pp_runs=("ball_runs", "sum"), pp_wickets=("is_dismissal", "sum"))
             .reset_index())

    result = {}
    for venue in inn["venue"].unique():
        if venue in ("Unknown", ""):
            continue
        vdf     = inn[inn["venue"] == venue]
        vdf_win = vdf[vdf["year"] >= primary_year - lookback_years + 1]
        vdf_use = vdf_win if len(vdf_win) >= MIN_MATCHES else vdf
        if len(vdf_use) < MIN_MATCHES:
            continue

        def _stats(df, inn_num):
            sub = df[df["inning"] == inn_num]["pp_runs"]
            if len(sub) < 2:
                return None
            return {"avg": float(sub.mean()), "lo": float(sub.quantile(0.25)),
                    "hi": float(sub.quantile(0.75)), "std": float(sub.std()),
                    "n":  int(len(sub))}

        pri1 = _stats(vdf[vdf["year"] == primary_year], 1)
        pri2 = _stats(vdf[vdf["year"] == primary_year], 2)
        win1 = _stats(vdf_use, 1)
        win2 = _stats(vdf_use, 2)
        if win1 is None:
            continue

        def _blend(pri, win, key):
            if pri is not None and pri["n"] >= MIN_MATCHES:
                return 0.60 * pri[key] + 0.40 * win[key]
            return win[key]

        i1_avg = _blend(pri1, win1, "avg")
        i2_avg = (_blend(pri2, win2, "avg") if pri2 and win2 else
                  (win2["avg"] if win2 else i1_avg * 0.95))
        wkts1        = vdf_use[vdf_use["inning"] == 1]["pp_wickets"]
        wkts_normal  = float(wkts1.mean()) if len(wkts1) else 1.5
        wkts_3plus_s = wkts1[wkts1 >= 3]
        wkts_3plus   = float(wkts_3plus_s.mean()) if len(wkts_3plus_s) >= 2 else i1_avg * 0.65

        vkey   = _get_venue_key(venue)
        nature = _VENUE_NATURE.get(vkey, "balanced") if vkey else _derive_nature(i1_avg)

        result[venue] = {
            "inn1_avg":         round(i1_avg, 2),
            "inn2_avg":         round(i2_avg, 2),
            "wkts_normal":      round(wkts_normal, 2),
            "wkts_3plus":       round(wkts_3plus, 2),
            "inn1_lo":          round(_blend(pri1, win1, "lo") * 0.70, 2),
            "inn1_hi":          round(_blend(pri1, win1, "hi") * 1.40, 2),
            "inn2_lo":          round((_blend(pri2, win2, "lo") if pri2 and win2 else i2_avg * 0.80) * 0.70, 2),
            "inn2_hi":          round((_blend(pri2, win2, "hi") if pri2 and win2 else i2_avg * 1.10) * 1.60, 2),
            "wkts_3plus_hi":    round(wkts_3plus * 1.10, 2),
            "inn1_spread":      round(_blend(pri1, win1, "hi") - _blend(pri1, win1, "lo"), 2),
            "inn2_spread":      round((_blend(pri2, win2, "hi") - _blend(pri2, win2, "lo")
                                       if pri2 and win2 else i2_avg * 0.20), 2),
            "nature":           nature,
            "n_matches_window": win1["n"] if win1 else 0,
        }

    _STADIUM_PP_DATA     = result
    _STADIUM_VENUE_INDEX = _build_venue_index(list(result.keys()))
    return result


def _load_deliveries(path=DELIVERIES_CSV_PATH):
    return pd.read_csv(path, low_memory=False)


def _load_matches(path=MATCHES_CSV_PATH):
    try:
        all_cols = pd.read_csv(path, nrows=0).columns.tolist()
        use = [c for c in MATCHES_USECOLS if c in all_cols]
        return pd.read_csv(path, usecols=use, low_memory=False)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════ #
#  MODEL                                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class MyModel:
    def __init__(self):
        self._is_fitted       = False
        self._id_to_name      = {}
        self._venue_index     = {}
        self._le              = {}
        self._latest_season   = 2026.0
        self._recent_cutoff   = 2024.0   # ↑ tighter recency window for 2026

        self._pp_global_avg   = 55.0
        self._pp_global_avg_r = 55.0
        self._pp_2025_avg     = None
        self._pp_2025_venue   = {}
        self._pp_2025_n_venue = {}

        self._k_venue   = 15
        self._k_team    = 10
        self._k_matchup = 8
        self._k_vt      = 6

        self._vi_avg = {}; self._vi_avg_r = {}; self._vi_n = {}
        self._ti_avg = {}; self._ti_avg_r = {}; self._ti_n = {}
        self._mu_avg = {}; self._mu_avg_r = {}; self._mu_n = {}
        self._vt_avg = {}; self._vt_avg_r = {}; self._vt_n = {}
        self._vtm_avg= {}; self._vtm_n    = {}
        self._vbowl_avg = {}; self._vbowl_avg_r = {}; self._vbowl_n = {}

        self._venue_trend_map = {}
        self._pp_inn1_avg  = {}; self._pp_inn1_avg_r = {}
        self._pp_inn2_avg  = {}; self._pp_inn2_avg_r = {}
        self._pp_inn_diff  = {}; self._pp_inn_diff_r = {}
        self._pp_global_diff   = 0.0
        self._pp_global_diff_r = 0.0

        self._team_pp_br = {}; self._team_pp_dr = {}
        self._team_vs_team_pp = {}

        self._bat_pp_avg   = {}; self._bat_pp_avg_r = {}
        self._bat_pp_sr    = {}; self._bat_pp_sr_r  = {}
        self._bat_pp_rr    = {}; self._bat_pp_rr_r  = {}
        self._bat_pp_n     = {}
        self._team_bat_median = {}

        self._bowl_pp_econ  = {}; self._bowl_pp_econ_r = {}
        self._bowl_pp_dot   = {}; self._bowl_pp_n      = {}
        self._bowl_pp_wkts  = {}; self._bowl_pp_wkts_n = {}
        self._team_bowl_econ   = {}; self._team_bowl_econ_r = {}
        self._team_bowl_median = {}

        self._team_rolling_bat  = {}
        self._team_rolling_bowl = {}
        self._h2h_pp_rundiff    = {}

        self._toss_inn2_adj      = 0.0
        self._toss_bat_first_adj = 0.0
        self._venue_bounds       = {}

        self._xgb  = None
        self._cat  = None
        self._et   = None
        self._gbm  = None
        self._meta = None

        self._n_train = 0

        # ── Playoff flag ─────────────────────────────────────────────────── #
        self._is_playoff = False

    # ─────────────────────────────────────────────────────────────────────── #
    #  FIT                                                                    #
    # ─────────────────────────────────────────────────────────────────────── #

    def fit(self, deliveries_df=None, players_df=None, matches_df=None):
        if deliveries_df is None:
            deliveries_df = _load_deliveries()
        if matches_df is None:
            matches_df = _load_matches()

        self._id_to_name = dict(_ID_TO_NAME)
        if players_df is not None:
            for _, r in players_df.iterrows():
                pid  = str(r.get("player_id", r.get("id", ""))).strip()
                name = str(r.get("delivery_name", r.get("name", ""))).strip()
                if pid and name:
                    self._id_to_name[pid] = name

        d = deliveries_df.copy()
        for col in ["batsman_runs", "extras"]:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)
        d["isWide"]   = pd.to_numeric(d.get("isWide",  0), errors="coerce").fillna(0)
        d["isNoBall"] = pd.to_numeric(d.get("isNoBall",0), errors="coerce").fillna(0)
        d["ball_runs"]   = d["batsman_runs"] + d["extras"]
        d["over"]        = pd.to_numeric(d.get("over", 0), errors="coerce").fillna(0).astype(int)
        d["year"]        = pd.to_datetime(d.get("date"), errors="coerce").dt.year.fillna(2020)
        d["is_legal"]    = ((d["isWide"] == 0) & (d["isNoBall"] == 0)).astype(int)
        d["is_dismissal"] = 0
        if "player_dismissed" in d.columns:
            d["is_dismissal"] = d["player_dismissed"].apply(
                lambda x: 0 if (pd.isna(x) or str(x).strip() in ("", "nan")) else 1)

        if matches_df is not None and "venue" in matches_df.columns:
            vm = (matches_df[["matchId", "venue"]].drop_duplicates("matchId")
                  .astype({"matchId": int, "venue": str}))
            d = d.merge(vm, on="matchId", how="left")

        if "venue" not in d.columns:
            d["venue"] = "Unknown"
        else:
            d["venue"] = d["venue"].fillna("Unknown")

        _compute_stadium_pp_data(d, matches_df)

        pp = d[d["over"] < 6].copy()

        grp_cols = ["matchId", "inning", "batting_team", "bowling_team",
                    "venue", "year"]
        grp_cols = [c for c in grp_cols if c in pp.columns]
        inn_pp = (pp.groupby(grp_cols, observed=True)
                    .agg(pp_runs    =("ball_runs",    "sum"),
                         pp_wickets =("is_dismissal", "sum"),
                         legal_balls=("is_legal",     "sum"))
                    .reset_index())
        inn_pp["year"] = pd.to_numeric(inn_pp["year"], errors="coerce").fillna(2020)

        self._n_train = len(inn_pp)

        self._pp_global_avg   = float(inn_pp["pp_runs"].mean())
        recent = inn_pp[inn_pp["year"] >= self._recent_cutoff]
        self._pp_global_avg_r = float(recent["pp_runs"].mean()) if len(recent) else self._pp_global_avg

        yr2025 = inn_pp[inn_pp["year"] == 2025]
        if len(yr2025) >= 5:
            self._pp_2025_avg = float(yr2025["pp_runs"].mean())

        for (venue,), grp in inn_pp.groupby(["venue"], observed=True):
            self._vi_avg[venue] = float(grp["pp_runs"].mean())
            self._vi_n[venue]   = len(grp)
            rec = grp[grp["year"] >= self._recent_cutoff]
            self._vi_avg_r[venue] = float(rec["pp_runs"].mean()) if len(rec) >= 3 else self._vi_avg[venue]
            v25 = grp[grp["year"] == 2025]
            if len(v25) >= 3:
                self._pp_2025_venue[venue]   = float(v25["pp_runs"].mean())
                self._pp_2025_n_venue[venue] = len(v25)

        for (venue, inn), grp in inn_pp.groupby(["venue", "inning"], observed=True):
            key = (venue, inn)
            self._pp_inn1_avg[key] = float(grp["pp_runs"].mean())
            rec = grp[grp["year"] >= self._recent_cutoff]
            self._pp_inn1_avg_r[key] = float(rec["pp_runs"].mean()) if len(rec) >= 3 else self._pp_inn1_avg[key]

        if "batting_team" in inn_pp.columns:
            for (team,), grp in inn_pp.groupby(["batting_team"], observed=True):
                self._ti_avg[team] = float(grp["pp_runs"].mean())
                self._ti_n[team]   = len(grp)
                rec = grp[grp["year"] >= self._recent_cutoff]
                self._ti_avg_r[team] = float(rec["pp_runs"].mean()) if len(rec) >= 3 else self._ti_avg[team]

        if "bowling_team" in inn_pp.columns:
            for (team,), grp in inn_pp.groupby(["bowling_team"], observed=True):
                self._vbowl_avg[team] = float(grp["pp_runs"].mean())
                self._vbowl_n[team]   = len(grp)
                rec = grp[grp["year"] >= self._recent_cutoff]
                self._vbowl_avg_r[team] = float(rec["pp_runs"].mean()) if len(rec) >= 3 else self._vbowl_avg[team]

        if "batting_team" in inn_pp.columns:
            for (venue, team), grp in inn_pp.groupby(["venue", "batting_team"], observed=True):
                self._vt_avg[(venue, team)] = float(grp["pp_runs"].mean())
                self._vt_n[(venue, team)]   = len(grp)
                rec = grp[grp["year"] >= self._recent_cutoff]
                self._vt_avg_r[(venue, team)] = float(rec["pp_runs"].mean()) if len(rec) >= 3 else self._vt_avg[(venue, team)]

        if "batting_team" in inn_pp.columns and "bowling_team" in inn_pp.columns:
            for (bt, bwt), grp in inn_pp.groupby(["batting_team", "bowling_team"], observed=True):
                self._mu_avg[(bt, bwt)] = float(grp["pp_runs"].mean())
                self._mu_n[(bt, bwt)]   = len(grp)
                rec = grp[grp["year"] >= self._recent_cutoff]
                self._mu_avg_r[(bt, bwt)] = float(rec["pp_runs"].mean()) if len(rec) >= 3 else self._mu_avg[(bt, bwt)]

        inn1 = inn_pp[inn_pp["inning"] == 1].set_index("matchId")["pp_runs"]
        inn2 = inn_pp[inn_pp["inning"] == 2].set_index("matchId")["pp_runs"]
        common = inn1.index.intersection(inn2.index)
        if len(common) >= 5:
            self._pp_global_diff = float((inn2.loc[common] - inn1.loc[common]).mean())
        recent_common = inn_pp[inn_pp["year"] >= self._recent_cutoff]
        ri1 = recent_common[recent_common["inning"] == 1].set_index("matchId")["pp_runs"]
        ri2 = recent_common[recent_common["inning"] == 2].set_index("matchId")["pp_runs"]
        rc = ri1.index.intersection(ri2.index)
        if len(rc) >= 5:
            self._pp_global_diff_r = float((ri2.loc[rc] - ri1.loc[rc]).mean())

        if matches_df is not None and "toss_decision" in matches_df.columns:
            try:
                tm = matches_df.copy()
                tm["matchId"] = pd.to_numeric(tm["matchId"], errors="coerce")
                inn_with_toss = inn_pp.merge(
                    tm[["matchId", "toss_winner", "toss_decision"]], on="matchId", how="left")
                bat_first = inn_with_toss[
                    (inn_with_toss["inning"] == 2) &
                    (inn_with_toss["toss_decision"].str.lower().str.strip() == "bat")]
                field_first = inn_with_toss[
                    (inn_with_toss["inning"] == 2) &
                    (inn_with_toss["toss_decision"].str.lower().str.strip() == "field")]
                if len(bat_first) >= 5 and len(field_first) >= 5:
                    self._toss_inn2_adj = float(
                        field_first["pp_runs"].mean() - bat_first["pp_runs"].mean())
            except Exception:
                pass

        for venue, grp in inn_pp.groupby("venue", observed=True):
            runs = grp["pp_runs"]
            self._venue_bounds[venue] = (
                float(runs.quantile(0.05)),
                float(runs.quantile(0.97)),
            )

        self._fit_ml_models(inn_pp)
        self._is_fitted = True

    # ─────────────────────────────────────────────────────────────────────── #
    #  ML MODEL FITTING                                                       #
    # ─────────────────────────────────────────────────────────────────────── #

    def _build_features(self, inn_pp: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame()
        feats["year"]        = inn_pp["year"].astype(float)
        feats["inning"]      = inn_pp["inning"].astype(float)
        feats["is_recent"]   = (inn_pp["year"] >= self._recent_cutoff).astype(float)
        feats["is_2025"]     = (inn_pp["year"] == 2025).astype(float)
        feats["is_2026"]     = (inn_pp["year"] == 2026).astype(float)

        if "batting_team" in inn_pp.columns:
            bat_enc = LabelEncoder()
            feats["bat_team_enc"] = bat_enc.fit_transform(
                inn_pp["batting_team"].fillna("Unknown"))
            self._le["bat"] = bat_enc
            feats["bat_hist_avg"] = inn_pp["batting_team"].map(
                lambda t: self._ti_avg.get(t, self._pp_global_avg))
            feats["bat_hist_avg_r"] = inn_pp["batting_team"].map(
                lambda t: self._ti_avg_r.get(t, self._pp_global_avg_r))
            feats["bat_correction"] = inn_pp["batting_team"].map(
                lambda t: _get_team_batting_correction(t))
            feats["bat_aggression"] = inn_pp["batting_team"].map(
                lambda t: _get_team_pp_aggression(t))
        else:
            feats["bat_team_enc"]   = 0.0
            feats["bat_hist_avg"]   = self._pp_global_avg
            feats["bat_hist_avg_r"] = self._pp_global_avg_r
            feats["bat_correction"] = 0.0
            feats["bat_aggression"] = 1.0

        if "bowling_team" in inn_pp.columns:
            bowl_enc = LabelEncoder()
            feats["bowl_team_enc"] = bowl_enc.fit_transform(
                inn_pp["bowling_team"].fillna("Unknown"))
            self._le["bowl"] = bowl_enc
            feats["bowl_quality"] = inn_pp["bowling_team"].map(
                lambda t: _get_team_pp_bowling_quality(t))
        else:
            feats["bowl_team_enc"] = 0.0
            feats["bowl_quality"]  = 1.0

        if "venue" in inn_pp.columns:
            v_enc = LabelEncoder()
            feats["venue_enc"] = v_enc.fit_transform(inn_pp["venue"].fillna("Unknown"))
            self._le["venue"] = v_enc
            feats["venue_avg"] = inn_pp["venue"].map(
                lambda v: self._vi_avg.get(v, self._pp_global_avg))
            feats["venue_avg_r"] = inn_pp["venue"].map(
                lambda v: self._vi_avg_r.get(v, self._pp_global_avg_r))
            feats["venue_2025_avg"] = inn_pp["venue"].map(
                lambda v: self._pp_2025_venue.get(v, self._pp_global_avg))
        else:
            feats["venue_enc"]      = 0.0
            feats["venue_avg"]      = self._pp_global_avg
            feats["venue_avg_r"]    = self._pp_global_avg_r
            feats["venue_2025_avg"] = self._pp_global_avg

        feats["recency_w"] = inn_pp["year"].map(_recency_weight)
        return feats

    def _fit_ml_models(self, inn_pp: pd.DataFrame):
        if len(inn_pp) < 30:
            return
        try:
            X = self._build_features(inn_pp)
            y = inn_pp["pp_runs"].values.astype(float)

            w = inn_pp["year"].map(_recency_weight).values
            w = w / w.sum() * len(w)

            tscv = TimeSeriesSplit(n_splits=3)

            self._xgb = xgb.XGBRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.07,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0,
                random_state=42, verbosity=0, n_jobs=1)
            self._xgb.fit(X, y, sample_weight=w)

            self._et = ExtraTreesRegressor(
                n_estimators=100, max_depth=6, min_samples_leaf=4,
                random_state=42, n_jobs=1)
            self._et.fit(X, y, sample_weight=w)

            self._gbm = HistGradientBoostingRegressor(
                max_iter=120, max_depth=5, learning_rate=0.07,
                min_samples_leaf=6, l2_regularization=1.0,
                random_state=42)
            self._gbm.fit(X, y)

            oof_xgb = np.zeros(len(y))
            oof_et  = np.zeros(len(y))
            oof_gbm = np.zeros(len(y))
            for tr_idx, val_idx in tscv.split(X):
                Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
                ytr = y[tr_idx]
                wtr = w[tr_idx]

                m_xgb = xgb.XGBRegressor(
                    n_estimators=100, max_depth=5, learning_rate=0.07,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=1.0, reg_lambda=2.0,
                    random_state=42, verbosity=0, n_jobs=1)
                m_xgb.fit(Xtr, ytr, sample_weight=wtr)
                oof_xgb[val_idx] = m_xgb.predict(Xval)

                m_et = ExtraTreesRegressor(
                    n_estimators=80, max_depth=6, min_samples_leaf=4,
                    random_state=42, n_jobs=1)
                m_et.fit(Xtr, ytr, sample_weight=wtr)
                oof_et[val_idx] = m_et.predict(Xval)

                m_gbm = HistGradientBoostingRegressor(
                    max_iter=100, max_depth=5, learning_rate=0.07,
                    min_samples_leaf=6, l2_regularization=1.0,
                    random_state=42)
                m_gbm.fit(Xtr, ytr)
                oof_gbm[val_idx] = m_gbm.predict(Xval)

            Xmeta = np.column_stack([oof_xgb, oof_et, oof_gbm])
            self._meta = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=1000)
            self._meta.fit(Xmeta, y)

        except Exception:
            pass

    def _predict_ml(self, row_feat: dict):
        if self._xgb is None or self._meta is None:
            return None
        try:
            X = self._build_features(pd.DataFrame([{
                "year":         row_feat.get("year", 2026),
                "inning":       row_feat.get("inning", 1),
                "batting_team": row_feat.get("batting_team", ""),
                "bowling_team": row_feat.get("bowling_team", ""),
                "venue":        row_feat.get("venue", ""),
            }]))
            for col in ["bat_team_enc", "bowl_team_enc", "venue_enc"]:
                if col not in X.columns:
                    X[col] = 0.0
            p_xgb = float(self._xgb.predict(X)[0])
            p_et  = float(self._et.predict(X)[0])
            p_gbm = float(self._gbm.predict(X)[0])
            Xm = np.array([[p_xgb, p_et, p_gbm]])
            return float(self._meta.predict(Xm)[0])
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────── #
    #  HEURISTIC PREDICTION                                                   #
    # ─────────────────────────────────────────────────────────────────────── #

    def _heuristic_predict(
        self,
        batting_team: str,
        bowling_team: str,
        venue: str,
        inning: int = 1,
        inn1_score=None,
        toss_winner=None,
        toss_decision=None,
        is_playoff: bool = False,
    ) -> float:
        sd = _get_stadium_data(venue)
        venue_key = _get_venue_key(venue)

        if sd is not None:
            base = sd["inn1_avg"] if inning == 1 else sd["inn2_avg"]
        else:
            base = _get_venue_prior(venue)

        v25 = self._pp_2025_venue.get(venue)
        if v25 is not None and self._pp_2025_n_venue.get(venue, 0) >= 3:
            base = 0.50 * base + 0.50 * v25

        base += _GLOBAL_2026_PP_OFFSET

        team_corr = _get_team_batting_correction(batting_team)

        away_adj = _away_team_pp_profile(batting_team, venue)
        if abs(team_corr) > 10:
            away_adj = away_adj * 0.50
        base += away_adj

        ti_r = self._ti_avg_r.get(batting_team, self._pp_global_avg_r)
        ti_n = self._ti_n.get(batting_team, 0)
        shrunk_ti = _shrink(ti_r, base, ti_n, k=self._k_team)
        base = 0.65 * base + 0.35 * shrunk_ti

        bq = _get_team_pp_bowling_quality(bowling_team)
        bowl_hist = self._vbowl_avg_r.get(bowling_team, self._pp_global_avg_r)
        bowl_adj = (bowl_hist - self._pp_global_avg_r) * 0.30
        base += bowl_adj * (2.0 - bq)

        vt_key = (venue, batting_team)
        if vt_key in self._vt_avg_r:
            vt_r = self._vt_avg_r[vt_key]
            vt_n = self._vt_n.get(vt_key, 0)
            base = _shrink(vt_r, base, vt_n, k=self._k_vt)

        mc1, mc2, mc_conf = _get_matchup_calibration(batting_team, bowling_team, venue)
        mc_adj = (mc1 if inning == 1 else mc2) * mc_conf
        base += mc_adj

        vk2 = venue_key or ""
        hb1, hb2, hb_conf = _get_home_pp_calibration(batting_team, vk2)
        home_boost = (hb1 if inning == 1 else hb2) * hb_conf
        base += home_boost

        nature = (sd["nature"] if sd else _VENUE_NATURE.get(vk2, "balanced"))
        ns = _effective_nature_score(vk2, nature, batting_team)
        aggression = _get_team_pp_aggression(batting_team)
        base += ns * aggression * 2.5

        venue_prior_raw = _get_venue_prior(venue)
        correction_weight = max(0.25, 0.55 - max(0.0, venue_prior_raw - 65.0) / 80.0)
        base += team_corr * correction_weight

        if inning == 2:
            base += self._pp_global_diff_r * 0.40
            chase_boost = _get_inn2_chase_boost(
                inn1_score if inn1_score is not None else base)
            base += chase_boost
            if inn1_score is not None:
                inn1_baseline = _get_venue_prior(venue) + _GLOBAL_2026_PP_OFFSET
                diff = inn1_score - inn1_baseline
                base += np.clip(diff * 0.18, -8, 12)
            # ── Dew bonus for night matches (all playoff venues) ────────── #
            dew_bonus = _get_dew_inn2_bonus(venue)
            base += dew_bonus

        if toss_winner and toss_decision:
            if (inning == 2 and
                    toss_decision.lower().strip() == "field" and
                    toss_winner.lower() in batting_team.lower()):
                base += self._toss_inn2_adj * 0.5

        # ── Playoff pressure discount ─────────────────────────────────────#
        # Knockout cricket: batters more cautious in PP; toss/conditions magnified
        if is_playoff:
            base += _PLAYOFF_PRESSURE_DISCOUNT

        if venue in self._venue_bounds:
            lo, hi = self._venue_bounds[venue]
            if base < lo:
                base = lo + (base - lo) * 0.3

        return float(max(base, 15.0))

    # ─────────────────────────────────────────────────────────────────────── #
    #  PREDICT — public API                                                   #
    # ─────────────────────────────────────────────────────────────────────── #

    def predict(self, test_data, is_playoff: bool = False):
        self._is_playoff = is_playoff
        if isinstance(test_data, pd.DataFrame):
            return self._predict_dataframe(test_data, is_playoff=is_playoff)
        if isinstance(test_data, dict):
            return self._predict_single(
                batting_team  = test_data.get("batting_team", ""),
                bowling_team  = test_data.get("bowling_team", ""),
                venue         = test_data.get("venue", ""),
                inning        = int(test_data.get("innings", test_data.get("inning", 1))),
                inn1_score    = test_data.get("inn1_score"),
                toss_winner   = test_data.get("toss_winner"),
                toss_decision = test_data.get("toss_decision"),
                is_playoff    = is_playoff,
            )
        raise TypeError(
            f"predict() expects a DataFrame or dict, got {type(test_data).__name__}.")

    def _predict_dataframe(self, test_data, is_playoff: bool = False):
        df = test_data.reset_index(drop=True)
        col_map = {c.lower().strip(): c for c in df.columns}

        def _col(candidates, default=None):
            for c in candidates:
                if c in col_map:
                    return df[col_map[c]]
            return pd.Series([default] * len(df))

        def _safe_na(v):
            try:
                return bool(pd.isna(v))
            except (TypeError, ValueError):
                return v is None

        ids           = _col(["id"])
        venues        = _col(["venue"],                                    default="")
        innings_s     = _col(["innings", "inning"],                        default=1)
        batting_teams = _col(["batting_team", "batting team", "bat_team"], default="")
        bowling_teams = _col(["bowling_team", "bowling team", "bowl_team"],default="")
        toss_w        = _col(["toss_winner",  "toss winner"],              default=None)
        toss_d        = _col(["toss_decision","toss decision"],            default=None)
        playoff_col   = _col(["is_playoff"],                               default=None)

        inn1_cache = {}
        rows = []

        for i in range(len(df)):
            bat   = "" if _safe_na(batting_teams.iloc[i]) else str(batting_teams.iloc[i]).strip()
            bowl  = "" if _safe_na(bowling_teams.iloc[i]) else str(bowling_teams.iloc[i]).strip()
            venue = "" if _safe_na(venues.iloc[i])        else str(venues.iloc[i]).strip()
            venue = validate_venue(venue)
            inn   = 1  if _safe_na(innings_s.iloc[i])     else int(innings_s.iloc[i])
            tw    = None if _safe_na(toss_w.iloc[i])      else str(toss_w.iloc[i])
            td    = None if _safe_na(toss_d.iloc[i])      else str(toss_d.iloc[i])

            # Row-level playoff override if column present
            row_playoff = is_playoff
            if not _safe_na(playoff_col.iloc[i]):
                row_playoff = bool(playoff_col.iloc[i])

            cache_key  = (venue.lower(), bowl.lower(), bat.lower())
            inn1_score = inn1_cache.get(cache_key)

            result   = self._predict_single(bat, bowl, venue, inn, inn1_score, tw, td,
                                             is_playoff=row_playoff)
            pred_val = result["predicted_score"]

            row_id = i if _safe_na(ids.iloc[i]) else ids.iloc[i]
            rows.append({"id": row_id, "predicted_score": pred_val})

            if inn == 1:
                inn1_cache[(venue.lower(), bat.lower(), bowl.lower())] = pred_val

        return pd.DataFrame(rows)

    def _predict_single(
        self,
        batting_team: str,
        bowling_team: str,
        venue: str,
        inning: int = 1,
        inn1_score=None,
        toss_winner=None,
        toss_decision=None,
        is_playoff: bool = False,
    ) -> dict:
        h = self._heuristic_predict(
            batting_team, bowling_team, venue, inning,
            inn1_score, toss_winner, toss_decision,
            is_playoff=is_playoff)

        ml_row = {
            "year":         2026,
            "inning":       inning,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue":        venue,
        }
        ml_pred = self._predict_ml(ml_row)

        if ml_pred is not None and self._n_train >= 50:
            ml_weight = min(0.35, 0.10 + 0.005 * (self._n_train - 50))
            pred = (1.0 - ml_weight) * h + ml_weight * ml_pred
            method = "ensemble"
        else:
            pred = h
            method = "heuristic"

        sd_data = _get_stadium_data(venue)
        if sd_data:
            spread = sd_data.get("inn1_spread" if inning == 1 else "inn2_spread", 20.0)
        else:
            spread = 22.0
        # Tighter CI in playoffs (less randomness with high stakes)
        if is_playoff:
            spread = spread * 0.90

        low  = max(15.0, pred - spread * 0.75)
        high = pred + spread * 0.90

        vk = _get_venue_key(venue)
        nature = sd_data["nature"] if sd_data else _VENUE_NATURE.get(vk or "", "balanced")

        n_venue   = self._vi_n.get(venue, 0)
        n_team    = self._ti_n.get(batting_team, 0)
        base_conf = min(0.85, 0.45 + 0.002 * n_venue + 0.001 * n_team)

        return {
            "predicted_score": round(pred, 1),
            "low":            round(low, 1),
            "high":           round(high, 1),
            "nature":         nature,
            "confidence":     round(base_conf, 2),
            "method":         method,
            "is_playoff":     is_playoff,
        }

    # ─────────────────────────────────────────────────────────────────────── #
    #  CONVENIENCE: predict both innings                                       #
    # ─────────────────────────────────────────────────────────────────────── #

    def predict_match(
        self,
        team1: str,
        team2: str,
        venue: str,
        toss_winner=None,
        toss_decision=None,
        is_playoff: bool = False,
    ) -> dict:
        inn1 = self._predict_single(
            batting_team=team1, bowling_team=team2,
            venue=venue, inning=1,
            toss_winner=toss_winner, toss_decision=toss_decision,
            is_playoff=is_playoff)

        inn2 = self._predict_single(
            batting_team=team2, bowling_team=team1,
            venue=venue, inning=2,
            inn1_score=inn1["predicted_score"],
            toss_winner=toss_winner, toss_decision=toss_decision,
            is_playoff=is_playoff)

        return {
            "inn1":      inn1,
            "inn2":      inn2,
            "venue":     venue,
            "team1":     team1,
            "team2":     team2,
            "is_playoff": is_playoff,
        }

if __name__ == "__main__":
    import sys, os

    TEST_CSV_PATH       = "/var/test_file.csv"
    SUBMISSION_CSV_PATH = "/var/submission.csv"
    LOGS_PATH           = "/var/logs.txt"

    def log(msg):
        print(msg)
        try:
            with open(LOGS_PATH, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    log("=== IPL PP Score Predictor ===")

    model = MyModel()
    model._is_fitted       = True
    model._n_train         = 0    # heuristic-only; no CSV data in container
    model._pp_global_avg   = 55.0
    model._pp_global_avg_r = 55.0

    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        log(f"Loaded test file: {len(test_df)} rows, columns: {list(test_df.columns)}")
    except Exception as e:
        log(f"ERROR loading test CSV: {e}")
        sys.exit(1)

    try:
        results = model.predict(test_df, is_playoff=False)
        log(f"Predictions complete: {len(results)} rows")
    except Exception as e:
        log(f"ERROR during prediction: {e}")
        sys.exit(1)

    try:
        results.to_csv(SUBMISSION_CSV_PATH, index=False)
        log(f"Submission written to {SUBMISSION_CSV_PATH}")
        log(results.to_string(index=False))
    except Exception as e:
        log(f"ERROR writing submission: {e}")
        sys.exit(1)
