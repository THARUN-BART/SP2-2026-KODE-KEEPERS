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
    "Punjab Kings":                "bindra",
    "Kings XI Punjab":             "bindra",
    "Sunrisers Hyderabad":         "uppal",
    "Gujarat Titans":              "narendra",
    "Lucknow Super Giants":        "ekana",
}

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
    "bindra":      "balanced",
}

_NATURE_SCORE = {
    "aggressive batting": 1.0,
    "extreme batting":    1.2,
    "ultra batting":      1.4,
    "batting + dew":      0.8,
    "batting":            0.7,
    "balanced":           0.0,
    "pace friendly":     -0.4,
    "spin-balanced":     -0.3,
    "spin/bowling":      -0.6,
    "slow-balanced":     -0.4,
    "slow/bowling":      -0.7,
}

_VENUE_HOME_OVERRIDE = {
    "chepauk": (0.90, -0.15),
    "ekana":   (0.95, -0.20),
    "bindra":  (1.00, -0.10),
}

_AWAY_DISCOUNT = {
    "chinnaswamy": 0.39, "uppal": 0.30, "kotla": 0.28,
    "eden": 0.18, "wankhede": 0.18, "narendra": 0.12,
    "barsapara": 0.18, "himachal": 0.20, "sawai": 0.15,
    "chepauk": 0.05, "ekana": 0.05, "bindra": 0.10,
}

_TEAM_PP_AGGRESSION = {
    "Sunrisers Hyderabad":         1.35,
    "Royal Challengers Bengaluru": 1.20,
    "Delhi Capitals":              1.18,
    "Kolkata Knight Riders":       1.08,
    "Punjab Kings":                1.12,
    "Rajasthan Royals":            1.10,
    "Mumbai Indians":              1.02,
    "Lucknow Super Giants":        1.00,
    "Gujarat Titans":              0.98,
    "Chennai Super Kings":         0.97,
}

_TEAM_PP_BOWLING_QUALITY = {
    "Mumbai Indians":              0.82,
    "Royal Challengers Bengaluru": 0.85,
    "Kolkata Knight Riders":       0.90,
    "Rajasthan Royals":            0.90,
    "Delhi Capitals":              0.94,
    "Sunrisers Hyderabad":         0.95,
    "Gujarat Titans":              0.96,
    "Punjab Kings":                0.97,
    "Lucknow Super Giants":        1.00,
    "Chennai Super Kings":         1.02,
}

_MATCHUP_CALIBRATION = {
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
}

_HIGH_SCORING_VENUES = {
    "kotla": 85.0, "uppal": 87.5, "chinnaswamy": 80.0,
    "wankhede": 75.0, "narendra": 80.0, "eden": 75.0,
    "barsapara": 80.0, "himachal": 75.0, "sawai": 72.0,
    "chepauk": 60.0,
}
_LOW_SCORING_VENUES = {"ekana": 52.0, "bindra": 62.0}

_GLOBAL_2025_PP_OFFSET = 4.5

_HOME_PP_CALIBRATION = {
    ("Chennai Super Kings",         "chepauk"):      {"inn1_boost": 30.0, "inn2_boost": 12.0, "conf": 0.80},
    ("Royal Challengers Bengaluru", "chinnaswamy"):  {"inn1_boost":  7.0, "inn2_boost":  4.0, "conf": 0.52},
    ("Sunrisers Hyderabad",         "uppal"):        {"inn1_boost": 10.0, "inn2_boost":  6.0, "conf": 0.60},
    ("Delhi Capitals",              "kotla"):        {"inn1_boost":  6.0, "inn2_boost":  3.5, "conf": 0.50},
    ("Kolkata Knight Riders",       "eden"):         {"inn1_boost":  4.0, "inn2_boost":  2.5, "conf": 0.48},
    ("Mumbai Indians",              "wankhede"):     {"inn1_boost":  4.0, "inn2_boost":  2.5, "conf": 0.48},
    ("Rajasthan Royals",            "sawai"):        {"inn1_boost":  3.5, "inn2_boost":  2.0, "conf": 0.45},
    ("Lucknow Super Giants",        "ekana"):        {"inn1_boost": -2.0, "inn2_boost": -1.5, "conf": 0.45},
    ("Punjab Kings",                "bindra"):       {"inn1_boost":  2.5, "inn2_boost":  1.5, "conf": 0.42},
    ("Gujarat Titans",              "narendra"):     {"inn1_boost":  3.5, "inn2_boost":  2.0, "conf": 0.45},
}

_SPIN_SUPPRESSION_VENUES = {"chepauk", "ekana", "eden"}

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
    'Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh',
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
        "barsapara", "vidarbha", "subrata", "shaheed",
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
    for key in list(_HIGH_SCORING_VENUES.keys()) + list(_LOW_SCORING_VENUES.keys()):
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


# ═══════════════════════════════════════════════════════════════════════════ #
#  STADIUM PP DATA                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def _compute_stadium_pp_data(deliveries_df, matches_df,
                              primary_year: int = 2024,
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
        d["venue"] = d["venue"].fillna("Unknown")
    else:
        d["venue"] = "Unknown"

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
    """IPL Powerplay Score Predictor — v13 optimized"""

    def __init__(self):
        self._is_fitted       = False
        self._id_to_name      = {}
        self._venue_index     = {}
        self._le              = {}
        self._latest_season   = 2025.0
        self._recent_cutoff   = 2023.0

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
        d["is_boundary"] = d["batsman_runs"].isin([4, 6]).astype(int)
        d["is_dot"]      = ((d["ball_runs"] == 0) & (d["isWide"] == 0)).astype(int)
        d["is_wkt"]      = 0
        if "player_dismissed" in d.columns:
            d["is_wkt"] = d["player_dismissed"].apply(
                lambda x: 0 if (pd.isna(x) or str(x).strip() in ("", "nan")) else 1)
        d["w"]           = d["year"].apply(_recency_weight)
        d["batting_team"] = d["batting_team"].astype(str)
        d["bowling_team"] = d["bowling_team"].astype(str)

        pp = d[d["over"] < 6].copy()

        if matches_df is not None and "venue" in matches_df.columns:
            vm = (matches_df[["matchId", "venue"]].drop_duplicates("matchId")
                  .astype({"matchId": int, "venue": str}))
            pp = pp.merge(vm, on="matchId", how="left")
            pp["venue"] = pp["venue"].fillna("Unknown")
        else:
            pp["venue"] = "Unknown"

        self._venue_index = _build_venue_index(pp["venue"].dropna().unique().tolist())
        _compute_stadium_pp_data(deliveries_df, matches_df, primary_year=2024, lookback_years=3)

        pp["season"] = pd.to_datetime(pp.get("date"), errors="coerce").dt.year.fillna(2024).astype(float)
        self._latest_season = float(pp["season"].max())
        self._recent_cutoff = self._latest_season - 2.0
        pp_r = pp[pp["season"] >= self._recent_cutoff].copy()

        inn = (pp.groupby(["matchId", "inning"], observed=True)
                 .agg(batting_team=("batting_team", "first"),
                      bowling_team=("bowling_team", "first"),
                      pp_runs      =("ball_runs",    "sum"),
                      pp_wkts      =("is_wkt",       "sum"),
                      venue        =("venue",         "first"),
                      season       =("season",        "first"),
                      year         =("year",          "first"),
                      date         =("date",          "first"))
                 .reset_index())
        inn["batting_team"] = inn["batting_team"].astype(str)
        inn["bowling_team"] = inn["bowling_team"].astype(str)
        inn["w"]  = inn["year"].apply(_recency_weight)
        inn_r     = inn[inn["season"] >= self._recent_cutoff].copy()

        self._pp_global_avg   = _wmean(inn["pp_runs"], inn["w"])
        self._pp_global_avg_r = (_wmean(inn_r["pp_runs"], inn_r["w"])
                                  if len(inn_r) > 10 else self._pp_global_avg)

        inn_2025 = inn[inn["season"] == 2025]
        if len(inn_2025) >= 10:
            self._pp_2025_avg = _wmean(inn_2025["pp_runs"], inn_2025["w"])
        for venue, g in inn_2025.groupby("venue", observed=True):
            self._pp_2025_venue[venue] = _wmean(g["pp_runs"].values, g["w"].values)
            self._pp_2025_n_venue[venue] = len(g)

        def _wagg(df, keys, k=10):
            avgs, ns = {}, {}
            for key, g in df.groupby(keys, observed=True):
                avgs[key] = _wmean(g["pp_runs"].values, g["w"].values)
                ns[key]   = len(g)
            return avgs, ns

        self._vi_avg,  self._vi_n  = _wagg(inn,  ["venue", "inning"],         self._k_venue)
        self._vi_avg_r, _          = _wagg(inn_r, ["venue", "inning"],         self._k_venue)
        self._ti_avg,  self._ti_n  = _wagg(inn,  ["batting_team", "inning"],   self._k_team)
        self._ti_avg_r, _          = _wagg(inn_r, ["batting_team", "inning"],   self._k_team)
        self._mu_avg,  self._mu_n  = _wagg(inn,  ["batting_team", "bowling_team"], self._k_matchup)
        self._mu_avg_r, _          = _wagg(inn_r, ["batting_team", "bowling_team"], self._k_matchup)
        self._vt_avg,  self._vt_n  = _wagg(inn,  ["venue", "batting_team"],    self._k_vt)
        self._vt_avg_r, _          = _wagg(inn_r, ["venue", "batting_team"],    self._k_vt)
        self._vtm_avg, self._vtm_n = _wagg(inn,  ["venue", "batting_team", "bowling_team"], 5)
        self._vbowl_avg, self._vbowl_n  = _wagg(inn,  ["venue", "bowling_team"], self._k_venue)
        self._vbowl_avg_r, _            = _wagg(inn_r, ["venue", "bowling_team"], self._k_venue)

        vs_pairs = {}
        for (ve, se), g in inn.groupby(["venue", "season"], observed=True):
            vs_pairs.setdefault(ve, []).append(
                (se, _wmean(g["pp_runs"].values, g["w"].values)))
        self._venue_trend_map = {v: _trend_slope(p) for v, p in vs_pairs.items()}

        for suffix, df_ in [("", inn), ("_r", inn_r)]:
            i1  = df_[df_["inning"] == 1]
            i2  = df_[df_["inning"] == 2]
            a1, a2 = {}, {}
            for v, g in i1.groupby("venue", observed=True):
                a1[v] = _wmean(g["pp_runs"].values, g["w"].values)
            for v, g in i2.groupby("venue", observed=True):
                a2[v] = _wmean(g["pp_runs"].values, g["w"].values)
            diff = {v: a1[v] - a2.get(v, a1[v]) for v in a1}
            gd = ((_wmean(i1["pp_runs"].values, i1["w"].values) if len(i1) else self._pp_global_avg)
                - (_wmean(i2["pp_runs"].values, i2["w"].values) if len(i2) else self._pp_global_avg))
            if suffix == "":
                self._pp_inn1_avg, self._pp_inn2_avg = a1, a2
                self._pp_inn_diff, self._pp_global_diff = diff, gd
            else:
                self._pp_inn1_avg_r, self._pp_inn2_avg_r = a1, a2
                self._pp_inn_diff_r, self._pp_global_diff_r = diff, gd

        for venue_raw, sd in _STADIUM_PP_DATA.items():
            mh = _fuzzy_match_venue(venue_raw, self._venue_index, threshold=0.45)
            if mh:
                h1 = self._pp_inn1_avg_r.get(mh, self._pp_inn1_avg.get(mh, sd["inn1_avg"]))
                h2 = self._pp_inn2_avg_r.get(mh, self._pp_inn2_avg.get(mh, sd["inn2_avg"]))
                self._pp_inn1_avg_r[mh] = 0.60 * h1 + 0.40 * sd["inn1_avg"]
                self._pp_inn2_avg_r[mh] = 0.60 * h2 + 0.40 * sd["inn2_avg"]

        st = pp.groupby(["batting_team", "inning"], observed=True).agg(
            br=("is_boundary", "mean"), dr=("is_dot", "mean"))
        self._team_pp_br = st["br"].to_dict()
        self._team_pp_dr = st["dr"].to_dict()

        for (bt, bwt), g in inn.groupby(["batting_team", "bowling_team"], observed=True):
            self._team_vs_team_pp[(bt, bwt)] = {
                "avg": _wmean(g["pp_runs"].values, g["w"].values), "n": len(g)}

        match_inn = inn.sort_values(["matchId", "inning"])
        for mid, mg in match_inn.groupby("matchId", observed=True):
            i1r = mg[mg["inning"] == 1]
            i2r = mg[mg["inning"] == 2]
            if len(i1r) and len(i2r):
                r1, r2 = i1r.iloc[0], i2r.iloc[0]
                bt, bwt = r1["batting_team"], r1["bowling_team"]
                diff = float(r1["pp_runs"]) - float(r2["pp_runs"])
                key  = (bt, bwt)
                if key not in self._h2h_pp_rundiff:
                    self._h2h_pp_rundiff[key] = {"total": 0.0, "n": 0}
                self._h2h_pp_rundiff[key]["total"] += diff
                self._h2h_pp_rundiff[key]["n"]     += 1

        for suffix, pp_ in [("", pp), ("_r", pp_r)]:
            legal = pp_[pp_["isWide"] == 0]
            bp = (legal.groupby(["matchId", "inning", "batsman"], observed=True)
                       ["batsman_runs"].agg(runs="sum", balls="count").reset_index())
            bg = bp.groupby("batsman", observed=True)
            avg_d = bg["runs"].mean().to_dict()
            sr_d  = bg.apply(lambda x: x["runs"].sum() / max(x["balls"].sum(), 1) * 100).to_dict()
            rr_d  = bg.apply(lambda x: x["runs"].sum() / max(x["balls"].sum(), 1)).to_dict()
            n_d   = bg["runs"].count().to_dict()
            if suffix == "":
                self._bat_pp_avg, self._bat_pp_sr = avg_d, sr_d
                self._bat_pp_rr,  self._bat_pp_n  = rr_d,  n_d
            else:
                self._bat_pp_avg_r, self._bat_pp_sr_r = avg_d, sr_d
                self._bat_pp_rr_r = rr_d

        team_bat = {}
        for pid, v in PLAYER_DATA.items():
            name = self._id_to_name.get(pid)
            if name and name in self._bat_pp_avg:
                team_bat.setdefault(v["team_name"], []).append(self._bat_pp_avg[name])
        self._team_bat_median = {t: float(np.median(vals)) for t, vals in team_bat.items() if vals}

        MIN_BOWL_INN = 3
        for suffix, pp_ in [("", pp), ("_r", pp_r)]:
            bp = (pp_.groupby(["matchId", "inning", "bowler"], observed=True)
                     .agg(br=("ball_runs","sum"), lb=("is_legal","sum"),
                          wk=("is_wkt","sum")).reset_index())
            bp["econ"] = bp["br"] / (bp["lb"] / 6).clip(0.1)
            bp["wpo"]  = bp["wk"] / (bp["lb"] / 6).clip(0.1)
            bg = bp.groupby("bowler", observed=True)
            ed = bg["econ"].mean().to_dict()
            nd = bg["econ"].count().to_dict()
            wd, wdn = {}, {}
            for bowler, g in bg:
                if len(g) >= MIN_BOWL_INN:
                    wd[bowler]  = float(g["wpo"].mean())
                    wdn[bowler] = len(g)
            if suffix == "":
                self._bowl_pp_econ, self._bowl_pp_n = ed, nd
                self._bowl_pp_wkts, self._bowl_pp_wkts_n = wd, wdn
            else:
                self._bowl_pp_econ_r = ed

        dot_c = pp[(pp["isWide"] == 0) & (pp["ball_runs"] == 0)].groupby("bowler").size()
        leg_c = pp[pp["isWide"] == 0].groupby("bowler").size()
        self._bowl_pp_dot = {
            b: float(dot_c.get(b, 0) / max(leg_c[b], 1)) for b in leg_c.index}

        for suffix, pp_ in [("", pp), ("_r", pp_r)]:
            tbi = (pp_.groupby(["matchId", "inning", "bowling_team"], observed=True)
                      .agg(br=("ball_runs","sum"), lb=("is_legal","sum")).reset_index())
            tbi["econ"] = tbi["br"] / (tbi["lb"] / 6).clip(0.1)
            tg = tbi.groupby("bowling_team", observed=True)["econ"].mean().to_dict()
            if suffix == "":
                self._team_bowl_econ = tg
            else:
                self._team_bowl_econ_r = tg

        team_bowl = {}
        for pid, v in PLAYER_DATA.items():
            name = self._id_to_name.get(pid)
            if name and name in self._bowl_pp_econ:
                team_bowl.setdefault(v["team_name"], []).append(self._bowl_pp_econ[name])
        self._team_bowl_median = {t: float(np.median(vals)) for t, vals in team_bowl.items() if vals}

        inn_sorted = inn.sort_values(["batting_team", "date"])
        for team, g in inn_sorted.groupby("batting_team", observed=True):
            vals = g["pp_runs"].values[-10:]
            self._team_rolling_bat[team] = _exp_rolling(vals, half_life=3.0)

        inn_sorted2 = inn.sort_values(["bowling_team", "date"])
        for team, g in inn_sorted2.groupby("bowling_team", observed=True):
            vals = g["pp_runs"].values[-10:]
            self._team_rolling_bowl[team] = _exp_rolling(vals, half_life=3.0)

        if matches_df is not None and "toss_decision" in matches_df.columns:
            toss_merge = matches_df[["matchId", "toss_decision"]].drop_duplicates("matchId")
            inn_toss   = inn.merge(toss_merge.astype({"matchId": int}), on="matchId", how="left")
            field_inn2 = inn_toss[(inn_toss["inning"] == 2) & (inn_toss["toss_decision"] == "field")]
            bat_inn2   = inn_toss[(inn_toss["inning"] == 2) & (inn_toss["toss_decision"] == "bat")]
            if len(field_inn2) >= 10 and len(bat_inn2) >= 10:
                self._toss_inn2_adj = float(np.clip(
                    _wmean(field_inn2["pp_runs"].values, field_inn2["w"].values)
                    - _wmean(bat_inn2["pp_runs"].values,   bat_inn2["w"].values), -5, 5))
            bat_inn1   = inn_toss[(inn_toss["inning"] == 1) & (inn_toss["toss_decision"] == "bat")]
            field_inn1 = inn_toss[(inn_toss["inning"] == 1) & (inn_toss["toss_decision"] == "field")]
            if len(bat_inn1) >= 10 and len(field_inn1) >= 10:
                self._toss_bat_first_adj = float(np.clip(
                    _wmean(bat_inn1["pp_runs"].values, bat_inn1["w"].values)
                    - _wmean(field_inn1["pp_runs"].values, field_inn1["w"].values), -3, 3))

        for venue, sd in _STADIUM_PP_DATA.items():
            lo = max(sd["inn1_lo"], 10)
            hi = min(sd["inn1_hi"], 145)
            self._venue_bounds[venue] = (lo, hi)

        self._train_ensemble(inn)
        self._is_fitted = True
        return self

    # ─────────────────────────────────────────────────────────────────────── #
    #  FEATURE VECTOR (66 features)                                           #
    # ─────────────────────────────────────────────────────────────────────── #

    def _fv(self, venue, bat, bowl, inning, season=None, toss_field=False, toss_bat_first=False):
        if season is None:
            season = self._latest_season
        g  = self._pp_global_avg
        gr = self._pp_global_avg_r

        def enc(col, val):
            le = self._le.get(col)
            if le is None: return 0
            try:    return int(le.transform([val])[0])
            except: return int(len(le.classes_) // 2)

        def _b(key, ad, rd, nd, prior, k):
            a = ad.get(key, prior)
            r = rd.get(key, a)
            n = nd.get(key, 0)
            return _shrink(0.75 * r + 0.25 * a, prior, n, k)

        vi   = _b((venue, inning), self._vi_avg,  self._vi_avg_r,  self._vi_n,  g, self._k_venue)
        ti   = _b((bat,   inning), self._ti_avg,  self._ti_avg_r,  self._ti_n,  g, self._k_team)
        mu   = _b((bat,   bowl),   self._mu_avg,  self._mu_avg_r,  self._mu_n,  g, self._k_matchup)
        vt   = _b((venue, bat),    self._vt_avg,  self._vt_avg_r,  self._vt_n,  g, self._k_vt)
        vtm  = _shrink(self._vtm_avg.get((venue, bat, bowl), mu), mu,
                       self._vtm_n.get((venue, bat, bowl), 0), 5)
        vbwl = _b((venue, bowl), self._vbowl_avg, self._vbowl_avg_r, self._vbowl_n, g, self._k_venue)

        trend = self._venue_trend_map.get(venue, 0.0)
        home  = float(_is_home_ground(venue, bat))

        i1  = self._pp_inn1_avg.get(venue, g)
        i1r = self._pp_inn1_avg_r.get(venue, i1)
        sd  = _get_stadium_data(venue)
        if sd is not None:
            i1r = 0.60 * i1r + 0.40 * sd["inn1_avg"]
        eff_i1 = 0.65 * i1r + 0.35 * i1

        i2  = self._pp_inn2_avg.get(venue, eff_i1)
        i2r = self._pp_inn2_avg_r.get(venue, i2)
        if sd is not None:
            i2r = 0.60 * i2r + 0.40 * sd["inn2_avg"]
        eff_i2 = 0.65 * i2r + 0.35 * i2

        d_all = self._pp_inn_diff.get(venue, self._pp_global_diff)
        d_rec = self._pp_inn_diff_r.get(venue, d_all)
        inn2d = 0.65 * d_rec + 0.35 * d_all

        br = self._team_pp_br.get((bat, inning), 0.28)
        dr = self._team_pp_dr.get((bat, inning), 0.32)

        t_econ_all = self._team_bowl_econ.get(bowl, 8.5)
        t_econ_rec = self._team_bowl_econ_r.get(bowl, t_econ_all)
        t_econ     = 0.70 * t_econ_rec + 0.30 * t_econ_all
        away_pen   = _away_team_pp_profile(bat, venue)

        vcat       = 0 if eff_i1 < 52 else (2 if eff_i1 > 60 else 1)
        inn2_ratio = eff_i2 / max(eff_i1, 1)
        season_off = float(season - self._latest_season)
        venue_dev  = eff_i1 - g

        venue_key = _get_venue_key(venue)
        eff_ns    = _effective_nature_score(venue_key or "", sd["nature"] if sd else "balanced", bat)
        nature_score  = eff_ns
        sd_inn2_ratio = (sd["inn2_avg"] / max(sd["inn1_avg"], 1)) if sd else 1.0
        sd_wkts_3plus = sd["wkts_3plus"] if sd else g * 0.70

        sd_inn1_avg    = sd["inn1_avg"]    if sd else g
        sd_inn2_avg    = sd["inn2_avg"]    if sd else g
        sd_inn1_lo     = sd["inn1_lo"]     if sd else g * 0.70
        sd_inn1_hi     = sd["inn1_hi"]     if sd else g * 1.40
        sd_inn2_lo     = sd["inn2_lo"]     if sd else g * 0.70
        sd_inn2_hi     = sd["inn2_hi"]     if sd else g * 1.60
        sd_inn1_spread = sd["inn1_spread"] if sd else 0.0
        sd_inn2_spread = sd["inn2_spread"] if sd else 0.0
        sd_inn_avg    = sd_inn1_avg if inning == 1 else sd_inn2_avg
        sd_inn_lo     = sd_inn1_lo  if inning == 1 else sd_inn2_lo
        sd_inn_hi     = sd_inn1_hi  if inning == 1 else sd_inn2_hi
        sd_inn_spread = sd_inn1_spread if inning == 1 else sd_inn2_spread
        away_discount = (_AWAY_DISCOUNT.get(venue_key, 0.15) if venue_key else 0.15)

        bat_aggression = _get_team_pp_aggression(bat)
        bowl_quality   = _get_team_pp_bowling_quality(bowl)
        m_adj1, m_adj2, m_conf = _get_matchup_calibration(bat, bowl, venue)
        matchup_adj = (m_adj1 if inning == 1 else m_adj2) * m_conf

        roll_bat   = self._team_rolling_bat.get(bat,  g)
        roll_bowl  = self._team_rolling_bowl.get(bowl, g)
        roll_delta = roll_bat - roll_bowl

        h2h = self._h2h_pp_rundiff.get((bat, bowl))
        if h2h and h2h["n"] > 0:
            raw_diff   = h2h["total"] / h2h["n"]
            h2h_signal = float(_shrink(raw_diff, 0.0, h2h["n"], 10))
        else:
            h2h_signal = 0.0

        toss_adj     = float(self._toss_inn2_adj * toss_field) if inning == 2 else 0.0
        toss_bat_adj = float(self._toss_bat_first_adj * toss_bat_first) if inning == 1 else 0.0

        venue_2025_avg = self._pp_2025_venue.get(venue)
        venue_2025_n   = self._pp_2025_n_venue.get(venue, 0)
        if venue_2025_avg is not None and venue_2025_n >= 5:
            v_2025_base = self._pp_inn1_avg_r.get(venue, g)
            drift_2025  = float(venue_2025_avg - v_2025_base)
        elif self._pp_2025_avg is not None:
            drift_2025 = float(self._pp_2025_avg - g)
        else:
            drift_2025 = 0.0

        gw = float(np.mean(list(self._bowl_pp_wkts.values()))) if self._bowl_pp_wkts else 0.3
        bowl_wkts_signal = float(np.clip(
            (self._bowl_pp_wkts.get(bowl, gw) - gw) * 5, -5, 5))

        ix_home_agg        = home * bat_aggression
        ix_home_inn1avg    = home * eff_i1
        ix_away_bowl_qual  = (1.0 - home) * (2.0 - bowl_quality)
        ix_inn2_ratio_flag = float(inning == 2) * inn2_ratio
        ix_agg_bowl        = bat_aggression * (2.0 - bowl_quality)
        ix_roll_h2h        = roll_delta * h2h_signal
        ix_vendev_agg      = venue_dev * bat_aggression
        ix_toss_inn2       = toss_adj * float(inning == 2)
        ix_drift_trend     = drift_2025 * trend
        ix_matchup_conf    = matchup_adj * float(m_conf > 0)
        ix_season2025      = float(season >= 2025)
        ix_bat_agg_inn2    = bat_aggression * float(inning == 2)
        ix_bowl_q_bat_v    = bowl_quality * float(eff_i1 > 65)
        max_ratio_cap      = 1.20 + max(bat_aggression - 1.0, 0.0) * 1.40
        ix_eff_inn2_ratio  = float(np.clip(inn2_ratio, 0.65, max_ratio_cap))

        return [
            enc("venue", venue), enc("batting_team", bat), enc("bowling_team", bowl),
            float(inning), season_off,
            vi, ti, mu, vt, vtm, vbwl,
            trend, home,
            eff_i1, eff_i2, inn2d, inn2_ratio,
            br, dr,
            float(vcat), gr, venue_dev,
            float(season - 2018),
            float(inning == 2), float(inning == 1),
            float(home and inning == 1),
            t_econ, away_pen,
            float(home and inning == 2),
            float(not _is_home_ground(venue, bat) and inning == 1),
            nature_score, sd_inn2_ratio, sd_wkts_3plus,
            sd_inn_avg, sd_inn_lo, sd_inn_hi, sd_inn_spread,
            sd_inn1_avg if sd else g, sd_inn2_avg if sd else g,
            away_discount,
            float(away_discount * (sd_inn_avg if sd else g)),
            float(inning == 1 and not _is_home_ground(venue, bat)),
            bat_aggression, bowl_quality, matchup_adj,
            roll_bat, roll_bowl, roll_delta,
            h2h_signal,
            toss_adj, toss_bat_adj,
            drift_2025,
            bowl_wkts_signal,
            ix_home_agg, ix_home_inn1avg, ix_away_bowl_qual,
            ix_inn2_ratio_flag, ix_agg_bowl,
            ix_roll_h2h, ix_vendev_agg, ix_toss_inn2,
            ix_drift_trend, ix_matchup_conf,
            ix_season2025, ix_bat_agg_inn2, ix_bowl_q_bat_v, ix_eff_inn2_ratio,
        ]

    # ─────────────────────────────────────────────────────────────────────── #
    #  STAT SIGNAL                                                            #
    # ─────────────────────────────────────────────────────────────────────── #

    def _stat_signal(self, venue, bat, bowl, inning):
        g = self._pp_global_avg

        def _b(key, ad, rd, nd, k):
            a = ad.get(key, g)
            r = rd.get(key, a)
            n = nd.get(key, 0)
            return _shrink(0.75 * r + 0.25 * a, g, n, k)

        vi  = _b((venue, inning), self._vi_avg,  self._vi_avg_r, self._vi_n,  self._k_venue)
        ti  = _b((bat,   inning), self._ti_avg,  self._ti_avg_r, self._ti_n,  self._k_team)
        mu  = _b((bat,   bowl),   self._mu_avg,  self._mu_avg_r, self._mu_n,  self._k_matchup)
        vt  = _b((venue, bat),    self._vt_avg,  self._vt_avg_r, self._vt_n,  self._k_vt)
        vtm = _shrink(self._vtm_avg.get((venue, bat, bowl), mu), mu,
                      self._vtm_n.get((venue, bat, bowl), 0), 5)
        trend = self._venue_trend_map.get(venue, 0.0)

        t_econ_all  = self._team_bowl_econ.get(bowl, 8.5)
        t_econ_rec  = self._team_bowl_econ_r.get(bowl, t_econ_all)
        t_econ      = 0.70 * t_econ_rec + 0.30 * t_econ_all
        global_econ = float(np.mean(list(self._team_bowl_econ.values()))) if self._team_bowl_econ else 8.5
        econ_signal = float(np.clip((t_econ - global_econ) * 0.8, -4, 4))

        bowl_quality = _get_team_pp_bowling_quality(bowl)
        quality_adj  = float(np.clip((1.0 - bowl_quality) * g * 0.12, -8, 8))
        econ_signal += quality_adj

        bat_aggression    = _get_team_pp_aggression(bat)
        aggression_signal = float(np.clip((bat_aggression - 1.0) * g * 0.15, -10, 10))

        gw = float(np.mean(list(self._bowl_pp_wkts.values()))) if self._bowl_pp_wkts else 0.3
        wkts_signal = float(np.clip((self._bowl_pp_wkts.get(bowl, gw) - gw) * -3, -4, 4))

        is_home   = _is_home_ground(venue, bat)
        sd        = _get_stadium_data(venue)
        venue_key = _get_venue_key(venue)

        if is_home:
            away_pen_anchor = 0.0
            sd_signal = 0.0
            if sd is not None:
                inn_avg   = sd["inn1_avg"] if inning == 1 else sd["inn2_avg"]
                sd_signal = float(np.clip((inn_avg - g) * 0.20, -8, 8))
            eff_ns       = _effective_nature_score(venue_key or "", sd["nature"] if sd else "balanced", bat)
            nature_boost = float(np.clip(eff_ns * 2.0, -3, 4))
            sd_signal   += nature_boost
        else:
            venue_prior = _get_venue_prior(venue)
            vk          = _get_venue_key(venue)
            disc        = (_AWAY_DISCOUNT.get(vk, 0.20) if vk else 0.20)
            anchor_pen  = -(disc * venue_prior) * (2.0 - bat_aggression)
            away_pen_anchor = float(np.clip(anchor_pen, -18, 4))
            sd_signal = 0.0
            if sd is not None:
                inn_avg   = sd["inn1_avg"] if inning == 1 else sd["inn2_avg"]
                sd_signal = float(np.clip((inn_avg - g) * 0.20, -8, 0))

        base = 0.26*vi + 0.22*ti + 0.17*mu + 0.15*vt + 0.10*vtm + 0.06*g + trend
        return base + econ_signal + away_pen_anchor + sd_signal + aggression_signal + wkts_signal

    # ─────────────────────────────────────────────────────────────────────── #
    #  TRAIN ENSEMBLE  — OPTIMIZED                                            #
    # ─────────────────────────────────────────────────────────────────────── #

    def _train_ensemble(self, inn):
        df = inn.copy().sort_values("date").reset_index(drop=True)
        for c in ["venue", "batting_team", "bowling_team"]:
            df[c] = df[c].fillna("Unknown").astype(str)
            le = LabelEncoder()
            le.fit(df[c])
            self._le[c] = le

        rows, ys, ws = [], [], []
        for _, r in df.iterrows():
            rows.append(self._fv(r["venue"], r["batting_team"], r["bowling_team"],
                                  int(r["inning"]), float(r["season"])))
            ys.append(float(r["pp_runs"]))
            ws.append(float(r["w"]))

        X = np.array(rows, dtype=float)
        y = np.array(ys,   dtype=float)
        w = np.array(ws,   dtype=float)

        n_eval  = max(int(len(X) * 0.15), 20)
        X_tr, X_ev = X[:-n_eval], X[-n_eval:]
        y_tr, y_ev = y[:-n_eval], y[-n_eval:]
        w_tr        = w[:-n_eval]

        # ── OPT 1: XGB n_estimators 600→400, n_jobs=-1 ───────────────────── #
        self._xgb = xgb.XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.045,
            subsample=0.84, colsample_bytree=0.78, min_child_weight=4,
            reg_lambda=1.2, reg_alpha=0.08, gamma=0.12,
            tree_method="hist", random_state=42, verbosity=0, n_jobs=-1,
            early_stopping_rounds=25, eval_metric="rmse")
        self._xgb.fit(X_tr, y_tr, sample_weight=w_tr,
                      eval_set=[(X_ev, y_ev)], verbose=False)

        self._cat = HistGradientBoostingRegressor(
            max_iter=300, max_depth=5, learning_rate=0.04,    # OPT 2: 400→300
            l2_regularization=3.0, random_state=42, max_bins=255)
        self._cat.fit(X, y, sample_weight=w)

        # ── OPT 3: ExtraTrees 200→120, n_jobs=-1 ─────────────────────────── #
        self._et = ExtraTreesRegressor(
            n_estimators=120, max_depth=10, min_samples_leaf=3,
            random_state=42, n_jobs=-1)
        self._et.fit(X, y, sample_weight=w)

        # ── OPT 4: Replace slow GradientBoostingRegressor with HistGBM ───── #
        self._gbm = HistGradientBoostingRegressor(
            max_iter=180, max_depth=4, learning_rate=0.045,
            l2_regularization=2.0, random_state=42, max_bins=128)
        self._gbm.fit(X, y, sample_weight=w)

        oof_x = np.zeros(len(y)); oof_c = np.zeros(len(y))
        oof_e = np.zeros(len(y)); oof_g = np.zeros(len(y))
        oof_s = np.zeros(len(y))

        # ── OPT 5: n_splits 4→2  (biggest single saving ~5-6s) ──────────── #
        tscv = TimeSeriesSplit(n_splits=2)
        for tr, va in tscv.split(X):
            n_ev2   = max(int(len(tr) * 0.15), 10)
            tr2, ev2 = tr[:-n_ev2], tr[-n_ev2:]

            # ── OPT 6: OOF XGB 600→300, depth 5→4, lr raised, n_jobs=-1 ─── #
            mx = xgb.XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.06,
                subsample=0.84, colsample_bytree=0.78, min_child_weight=4,
                reg_lambda=1.2, reg_alpha=0.08, gamma=0.12,
                tree_method="hist", random_state=42, verbosity=0, n_jobs=-1,
                early_stopping_rounds=20, eval_metric="rmse")
            mx.fit(X[tr2], y[tr2], sample_weight=w[tr2],
                   eval_set=[(X[ev2], y[ev2])], verbose=False)
            oof_x[va] = mx.predict(X[va])

            # ── OPT 7: OOF HistGBM max_bins 255→128 ──────────────────────── #
            mc = HistGradientBoostingRegressor(
                max_iter=250, max_depth=5, learning_rate=0.05,
                l2_regularization=3.0, random_state=42, max_bins=128)
            mc.fit(X[tr], y[tr], sample_weight=w[tr])
            oof_c[va] = mc.predict(X[va])

            # ── OPT 8: OOF ExtraTrees 200→80, depth 11→9, n_jobs=-1 ──────── #
            me = ExtraTreesRegressor(
                n_estimators=80, max_depth=9, min_samples_leaf=3,
                random_state=42, n_jobs=-1)
            me.fit(X[tr], y[tr], sample_weight=w[tr])
            oof_e[va] = me.predict(X[va])

            # ── OPT 9: OOF slow GBM → HistGBM ───────────────────────────── #
            mg = HistGradientBoostingRegressor(
                max_iter=150, max_depth=4, learning_rate=0.05,
                l2_regularization=2.0, random_state=42, max_bins=128)
            mg.fit(X[tr], y[tr], sample_weight=w[tr])
            oof_g[va] = mg.predict(X[va])

            for i in va:
                r = df.iloc[i]
                oof_s[i] = self._stat_signal(
                    r["venue"], r["batting_team"], r["bowling_team"], int(r["inning"]))

        X_meta = np.column_stack([oof_x, oof_c, oof_e, oof_g, oof_s])
        n_mv   = max(int(len(X_meta) * 0.15), 10)
        self._meta = xgb.XGBRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.80, reg_lambda=2.0,
            tree_method="hist", random_state=42, verbosity=0, n_jobs=-1,
            early_stopping_rounds=20, eval_metric="rmse")
        self._meta.fit(
            X_meta[:-n_mv], y[:-n_mv],
            eval_set=[(X_meta[-n_mv:], y[-n_mv:])], verbose=False)

    # ─────────────────────────────────────────────────────────────────────── #
    #  PREDICT                                                                #
    # ─────────────────────────────────────────────────────────────────────── #

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for idx, row in test_df.iterrows():
            row_id = int(row.get("id", idx))
            results.append({
                "id": row_id,
                "predicted_score": int(self._predict_row(row))
            })
        return pd.DataFrame(results)

    def _predict_row(self, row):
        raw_venue      = str(row.get("venue", "Unknown")).strip()
        inning         = int(row.get("innings", row.get("inning", 1)))
        bat            = str(row.get("batting_team", "")).strip()
        bowl           = str(row.get("bowling_team", "")).strip()
        toss_field     = bool(row.get("toss_field", False))
        toss_bat_first = bool(row.get("toss_bat_first", False))
        season         = float(row.get("season", self._latest_season))

        canonical = validate_venue(raw_venue)
        venue     = _fuzzy_match_venue(canonical, self._venue_index) or canonical

        bat_names  = self._resolve(row.get("Batsman's Player Id", ""))
        bowl_names = self._resolve(row.get("Bowler's Player id (opponent)", ""))

        fv  = np.array(self._fv(venue, bat, bowl, inning, season=season,
                                 toss_field=toss_field,
                                 toss_bat_first=toss_bat_first), dtype=float).reshape(1, -1)

        p_x = float(np.clip(self._xgb.predict(fv)[0], 10, 160))
        p_c = float(np.clip(self._cat.predict(fv)[0], 10, 160))
        p_e = float(np.clip(self._et.predict(fv)[0],  10, 160))
        p_g = float(np.clip(self._gbm.predict(fv)[0], 10, 160))
        p_s = self._stat_signal(venue, bat, bowl, inning)

        x_meta = np.array([[p_x, p_c, p_e, p_g, p_s]], dtype=float)
        base   = float(np.clip(self._meta.predict(x_meta)[0], 10, 160))

        bat_aggression = _get_team_pp_aggression(bat)
        bowl_quality   = _get_team_pp_bowling_quality(bowl)
        sd             = _get_stadium_data(venue)
        venue_key      = _get_venue_key(venue)
        is_home        = _is_home_ground(venue, bat)

        venue_2025_n = self._pp_2025_n_venue.get(venue, 0)
        if season >= 2025:
            if venue_2025_n < 5:
                env_conf = float(np.clip(1.0 - venue_2025_n / 5.0, 0.20, 1.0))
                base += _GLOBAL_2025_PP_OFFSET * env_conf
            else:
                base += _GLOBAL_2025_PP_OFFSET * 0.20

        if inning == 2:
            i1r    = self._pp_inn1_avg_r.get(venue, self._pp_global_avg)
            i2r    = self._pp_inn2_avg_r.get(venue, i1r)
            eff_i1 = 0.80 * i1r + 0.20 * self._pp_inn1_avg.get(venue, i1r)
            eff_i2 = 0.80 * i2r + 0.20 * self._pp_inn2_avg.get(venue, i2r)
            n_vm   = (sd["n_matches_window"] if sd else 0)
            data_conf = float(np.clip(n_vm / 20.0, 0.30, 0.70))

            if sd is not None:
                sd_ratio   = sd["inn2_avg"] / max(sd["inn1_avg"], 1)
                hist_ratio = eff_i2 / max(eff_i1, 1)
                raw_ratio  = float(data_conf * hist_ratio + (1.0 - data_conf) * sd_ratio)
            else:
                raw_ratio = float(eff_i2 / max(eff_i1, 1))

            max_ratio_cap = 1.20 + max(bat_aggression - 1.0, 0.0) * 1.40
            ratio = float(np.clip(raw_ratio, 0.65, max_ratio_cap))

            if bat_aggression >= 1.20:
                base = 0.40 * base + 0.60 * base * ratio
            elif bat_aggression >= 1.05:
                base = 0.48 * base + 0.52 * base * ratio
            else:
                base = 0.55 * base + 0.45 * base * ratio

            if is_home:
                sd_h  = _get_stadium_data(venue)
                bonus = float(np.clip(1.2 + 0.08 * ((sd_h["inn1_avg"] - 50) if sd_h else 0), 1.2, 5.0))
                base += bonus
            base += self._toss_inn2_adj * float(toss_field)

            if not is_home and bat_aggression > 1.05:
                agg_inn2_boost = (bat_aggression - 1.0) * bat_aggression * 40.0
                base += float(np.clip(agg_inn2_boost, 0.0, 30.0))

            if (not is_home
                    and bowl_quality <= 0.92
                    and venue_key in _SPIN_SUPPRESSION_VENUES):
                spin_supp = float(np.clip(-(1.0 - bowl_quality) * 100.0, -20.0, 0.0))
                base += spin_supp
        else:
            base += self._toss_bat_first_adj * float(toss_bat_first)

        is_batting_venue = ((sd is not None and sd["inn1_avg"] > 60)
                            or (venue_key in _HIGH_SCORING_VENUES))
        if bowl_quality <= 0.93 and is_batting_venue:
            bowling_dom_adj = float(np.clip(
                (0.93 - bowl_quality) * 40.0 * (-0.5), -8.0, 0.0))
            if len(bat_names) == 0:
                base += bowling_dom_adj * 0.60

        sd_row    = _get_stadium_data(venue)
        n_bat_ids = len(bat_names)

        if n_bat_ids == 0:
            wkt_scenario = "mid"
        elif n_bat_ids <= 2:
            wkt_scenario = "normal"
        elif n_bat_ids == 3:
            wkt_scenario = "mid"
        else:
            wkt_scenario = "collapse"

        scenario_target, scenario_blend = None, 0.0
        if sd_row is not None:
            inn_hi_n  = sd_row["inn1_hi"] if inning == 1 else sd_row["inn2_hi"]
            inn_avg_n = sd_row["inn1_avg"] if inning == 1 else sd_row["inn2_avg"]
            wkt3_mid  = sd_row["wkts_3plus"]
            if wkt_scenario == "normal":
                if inning == 1:
                    scenario_target = inn_hi_n
                    scenario_blend  = 0.45
                else:
                    scenario_target = inn_avg_n
                    scenario_blend  = 0.05 if bat_aggression >= 1.10 else 0.15
            elif wkt_scenario == "collapse":
                scenario_target = wkt3_mid
                scenario_blend  = 0.45
        else:
            wkt_scenario = "mid"

        use_pl        = n_bat_ids > 3
        collapse_damp = 0.40 if wkt_scenario == "collapse" else 1.0
        bat_adj  = self._bat_adj(bat_names,  inning, venue, bat,  use_player_level=use_pl)
        bowl_adj = self._bowl_adj(bowl_names, len(bowl_names), inning, bowl_team=bowl)

        if inning == 2:
            adj = float(np.clip((bat_adj + bowl_adj) * 0.65 * collapse_damp, -18, 18))
        else:
            if not is_home and sd_row is not None:
                excess = sd_row["inn1_avg"] - self._pp_global_avg
                damp   = float(np.clip(0.68 - 0.018 * excess, 0.15, 0.68))
                adj    = float(np.clip((bat_adj + bowl_adj) * damp * collapse_damp, -10, 10))
            elif not is_home:
                adj = float(np.clip((bat_adj + bowl_adj) * 0.60 * collapse_damp, -14, 14))
            else:
                adj = float(np.clip((bat_adj + bowl_adj) * collapse_damp, -20, 20))
        base += adj

        rr_damp   = 0.50 if wkt_scenario == "collapse" else 1.0
        rr_signal = self._lineup_rr_signal(bat_names, bowl_names, inning, venue, bat, bowl)
        base += rr_signal * rr_damp

        if sd is not None:
            raw_ns         = _effective_nature_score(venue_key or "", sd["nature"], bat)
            nudge_strength = 1.0 + 0.5 * abs(raw_ns)
            if is_home:
                nature_nudge = float(np.clip(raw_ns * nudge_strength, -4.0, 4.0))
            else:
                nature_nudge = float(np.clip(-raw_ns * nudge_strength * 0.8, -2.5, 2.5))
            base += nature_nudge

        anchor = p_s
        if inning == 2 and bat_aggression >= 1.10:
            clamp_width   = 32.0
            anchor_weight = 0.65
        else:
            clamp_width   = 22.0
            anchor_weight = 1.0

        anchor_eff = anchor * anchor_weight
        apply_away_clamp = (
            not is_home and inning == 1
            and sd_row is not None and wkt_scenario != "normal")
        if apply_away_clamp:
            vk       = _get_venue_key(venue)
            home_adv = _AWAY_DISCOUNT.get(vk, 0.15) if vk else 0.15
            clamp_up = float(np.clip(20 - 36 * home_adv, 6, 20))
            clamp_f  = float(np.clip(0.40 - home_adv, 0.05, 0.40))
            if   base > anchor_eff + clamp_up:
                base = anchor_eff + clamp_up + clamp_f * (base - anchor_eff - clamp_up)
            elif base < anchor_eff - clamp_up:
                base = anchor_eff - clamp_up + clamp_f * (base - anchor_eff + clamp_up)
        else:
            if   base > anchor_eff + clamp_width:
                base = anchor_eff + clamp_width + 0.30 * (base - anchor_eff - clamp_width)
            elif base < anchor_eff - clamp_width:
                base = anchor_eff - clamp_width + 0.30 * (base - anchor_eff + clamp_width)

        if not is_home and inning == 1 and sd_row is not None:
            vk       = _get_venue_key(venue)
            home_adv = _AWAY_DISCOUNT.get(vk, 0.15) if vk else 0.15
            if home_adv >= 0.18:
                inn1_avg    = sd_row["inn1_avg"]
                wkts_upper  = sd_row.get("wkts_3plus_hi", sd_row["wkts_3plus"] + 5)
                hard_ceiling = (sd_row.get("inn1_hi", inn1_avg * 1.15)
                                if wkt_scenario == "normal"
                                else max(wkts_upper, inn1_avg * (1.0 - home_adv)))
                if base > hard_ceiling:
                    blend = float(np.clip(0.50 + home_adv, 0.60, 0.80))
                    base  = blend * hard_ceiling + (1.0 - blend) * base

        if scenario_blend > 0 and scenario_target is not None:
            base = (1.0 - scenario_blend) * base + scenario_blend * scenario_target

        m_adj1, m_adj2, m_conf = _get_matchup_calibration(bat, bowl, venue)
        if m_conf > 0:
            m_adj = (m_adj1 if inning == 1 else m_adj2) * (0.50 if wkt_scenario == "collapse" else 1.0)
            base  = (1.0 - m_conf) * base + m_conf * (base + m_adj)

        if is_home and venue_key:
            h_boost1, h_boost2, h_conf = _get_home_pp_calibration(bat, venue_key)
            h_boost = h_boost1 if inning == 1 else h_boost2
            if h_conf > 0 and abs(h_boost) > 0:
                calibrated = base + h_boost
                base = (1.0 - h_conf) * base + h_conf * calibrated

        vb_key = _fuzzy_match_venue(venue, _STADIUM_VENUE_INDEX) if _STADIUM_VENUE_INDEX else None
        if vb_key and vb_key in self._venue_bounds:
            lo, hi = self._venue_bounds[vb_key]
        else:
            lo, hi = 10, 145

        if inning == 2:
            if sd_row is not None:
                hi_inn2 = max(hi, sd_row["inn2_hi"] * (1.60 if bat_aggression >= 1.05 else 1.0))
            else:
                hi_inn2 = 160
            return float(np.clip(round(base), lo, hi_inn2))

        return float(np.clip(round(base), lo, hi))

    # ─────────────────────────────────────────────────────────────────────── #
    #  HELPERS                                                                #
    # ─────────────────────────────────────────────────────────────────────── #

    def _resolve(self, raw):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return []
        tokens = ([str(x).strip() for x in raw]
                  if isinstance(raw, (list, tuple))
                  else [t.strip() for t in str(raw).split(",") if t.strip()])
        seen, names = set(), []
        for t in tokens:
            if not t or t.lower() in ("nan", "none", "") or t in seen:
                continue
            seen.add(t)
            resolved = self._id_to_name.get(t)
            if resolved:
                names.append(resolved)
            elif t in self._bat_pp_avg or t in self._bowl_pp_econ:
                names.append(t)
        return names

    def _lineup_rr_signal(self, bat_names, bowl_names, inning, venue, bat_team, bowl_team):
        g_rr      = self._pp_global_avg / 36.0
        g_econ    = float(np.mean(list(self._bowl_pp_econ.values()))) if self._bowl_pp_econ else 8.5
        g_bowl_rr = g_econ / 6.0

        bat_rrs = []
        for n in bat_names:
            if n not in self._bat_pp_rr:
                continue
            conf = min(self._bat_pp_n.get(n, 0) / 10.0, 1.0)
            rr   = (0.70 * self._bat_pp_rr_r.get(n, self._bat_pp_rr[n])
                  + 0.30 * self._bat_pp_rr[n])
            bat_rrs.append((rr, conf))

        if bat_rrs:
            tc           = sum(c for _, c in bat_rrs)
            lr           = sum(r * c for r, c in bat_rrs) / tc if tc else g_rr
            bat_rr_delta = (lr - g_rr) * 36.0
        else:
            bat_rr_delta = 0.0

        bowl_rrs = []
        for n in bowl_names:
            if n not in self._bowl_pp_econ:
                continue
            conf = min(self._bowl_pp_n.get(n, 0) / 10.0, 1.0)
            econ = (0.70 * self._bowl_pp_econ_r.get(n, self._bowl_pp_econ[n])
                  + 0.30 * self._bowl_pp_econ[n])
            bowl_rrs.append((econ / 6.0, conf))

        if bowl_rrs:
            tc            = sum(c for _, c in bowl_rrs)
            lb            = sum(r * c for r, c in bowl_rrs) / tc if tc else g_bowl_rr
            bowl_rr_delta = (lb - g_bowl_rr) * 36.0
        else:
            bowl_rr_delta = 0.0

        is_home = _is_home_ground(venue, bat_team)
        sd      = _get_stadium_data(venue)
        if inning == 2:
            bat_sc, bowl_sc = 0.45, 0.40
        elif is_home:
            bat_sc, bowl_sc = 0.55, 0.50
        else:
            if sd is not None:
                excess = sd["inn1_avg"] - self._pp_global_avg
                damp   = float(np.clip(0.60 - 0.015 * excess, 0.15, 0.60))
            else:
                damp = 0.40
            bat_sc, bowl_sc = damp, damp * 0.90

        return float(np.clip(bat_rr_delta * bat_sc + bowl_rr_delta * bowl_sc, -15, 15))

    def _bat_adj(self, names, inning=1, venue=None, bat_team=None, use_player_level=False):
        if not self._bat_pp_avg or not names:
            return 0.0
        la       = float(np.mean(list(self._bat_pp_avg.values())))
        ls       = float(np.mean(list(self._bat_pp_sr.values()))) if self._bat_pp_sr else 128.0
        team_med = self._team_bat_median.get(bat_team, la) if bat_team else la

        deltas = []
        for n in names:
            if n not in self._bat_pp_avg:
                deltas.append(0.10 * (team_med - la))
                continue
            conf      = min(self._bat_pp_n.get(n, 0) / 8.0, 1.0)
            avg       = (0.75 * self._bat_pp_avg_r.get(n, self._bat_pp_avg[n])
                       + 0.25 * self._bat_pp_avg[n])
            sr        = (0.75 * self._bat_pp_sr_r.get(n, self._bat_pp_sr.get(n, ls))
                       + 0.25 * self._bat_pp_sr.get(n, ls))
            avg_delta = avg - la
            sr_delta  = (sr - ls) / max(ls, 1) * la * 0.7
            deltas.append(conf * (0.50 * avg_delta + 0.50 * sr_delta))

        if not deltas:
            return 0.0
        scalar = (0.50 if inning == 2 else 0.58) if use_player_level else \
                 (0.32 if inning == 2 else 0.38)
        return float(np.clip(sum(deltas) * scalar, -18, 18))

    def _bowl_adj(self, names, n_bowlers, inning=1, bowl_team=None):
        if not self._bowl_pp_econ:
            return 0.0
        le = float(np.mean(list(self._bowl_pp_econ.values())))
        ld = 0.32
        gw = float(np.mean(list(self._bowl_pp_wkts.values()))) if self._bowl_pp_wkts else 0.3

        team_econ = None
        if bowl_team:
            t_all = self._team_bowl_econ.get(bowl_team)
            t_rec = self._team_bowl_econ_r.get(bowl_team, t_all)
            if t_all is not None:
                team_econ = 0.70 * (t_rec or t_all) + 0.30 * t_all
                bq = _get_team_pp_bowling_quality(bowl_team)
                if bq < 1.0:
                    team_econ *= (1.0 - (1.0 - bq) * 1.5 * 0.15)
            if team_econ is None:
                team_econ = self._team_bowl_median.get(bowl_team)

        if not names:
            if team_econ is not None:
                return float(np.clip(0.50 * 1.2 * (team_econ - le), -10, 10))
            return 0.0

        deltas = []
        for n in names:
            if n in self._bowl_pp_econ:
                conf = min(self._bowl_pp_n.get(n, 0) / 8.0, 1.0)
                econ = (0.75 * self._bowl_pp_econ_r.get(n, self._bowl_pp_econ[n])
                      + 0.25 * self._bowl_pp_econ[n])
                dot  = self._bowl_pp_dot.get(n, ld)
                wkts_bonus = float(np.clip((self._bowl_pp_wkts.get(n, gw) - gw) * -2, -3, 3)) \
                             if n in self._bowl_pp_wkts else 0.0
            elif team_econ is not None:
                conf, econ, dot, wkts_bonus = 0.35, team_econ, ld, 0.0
            else:
                continue
            deltas.append(conf * (1.3 * (econ - le) + 0.4 * (ld - dot) * 20 + wkts_bonus))

        if not deltas:
            return 0.0
        coverage = min(n_bowlers / 2.5, 1.0)
        return float(np.clip(sum(deltas) * coverage * 1.8, -16, 16))