
# coding: utf-8

#import libraries: 
import os
import xml.etree.cElementTree as ET
from collections import defaultdict
import pprint
import re
import codecs
import json
import string
import codecs
from pymongo import MongoClient
from datetime import datetime

# Read through the dataset with path.join function
filename = "DC.osm" 
path = "/Users/fengxinlin/Downloads" 
DC_OSM = os.path.join(path, filename)

#  Count the number of unique entry types
def count_tags(filename):
        tags = {}
        for event, elem in ET.iterparse(filename):
            if elem.tag in tags: 
                tags[elem.tag] += 1
            else:
                tags[elem.tag] = 1
        return tags
DC_tags = count_tags(DC_OSM)
pprint.pprint(DC_tags)

# check the "k" value for each entry.
lower = re.compile(r'^([a-z])*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
    
def key_type(element, keys):
    if element.tag == "tag":
        for tag in element.iter('tag'):
            k = tag.get('k')
            if lower.search(k):
                keys['lower'] += 1
            elif lower_colon.search(k):
                keys['lower_colon'] += 1
            elif problemchars.search(k):
                keys['problemchars'] += 1
            else:
                keys['other'] += 1
    return keys
  
def process_map(filename):
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)

    return keys

map_keys = process_map(DC_OSM)
pprint.pprint(map_keys)

# how many unique users  
def process_map(filename):
    users = set()
    for _, element in ET.iterparse(filename):
        for e in element:
            if 'uid' in e.attrib:
                users.add(e.attrib['uid'])
    return users
users = process_map(DC_OSM)
len(users)


# Wrangle  Street name
street_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
expected = ["Avenue", "Boulevard", "Commons", "Court", "Drive", "Lane", "Parkway", "Place", "Road", "Square", "Street",
            "Trail"]
mapping = {'Ave': 'Avenue', 'Blvd' : 'Boulevard', 'Dr' : 'Drive', 'Ln' : 'Lane', 'Pkwy' : 'Parkway', 'Rd' : 'Road', 
           'Rd.' : 'Road', 'St' : 'Street', 'street' :"Street", 'Ct' : "Court", 'Cir': "Circle", 'Cr' : "Court", 
           'ave' : 'Avenue', 'Hwg' : 'Highway', 'Hwy' : 'Highway', 'Sq' : "Square"}


# The "audit_street_type" function search the input string for its regex. If it matches and it is not within the "expected" list, add the match as a key and add the string to the set.
# The "is_street_name" function runs when k="addre:street" and looks at the attribute k.
# The "audit" function will return a list that match previous two functions. 

def audit_street_type(street_types, street_name):
    m = street_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)

            
def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    return street_types

DC_street_types = audit(DC_OSM)

# Print it out to have a brief look at the unclean street data set:
pprint.pprint(dict(DC_street_types))

# The function "update_name" for street rename processing.
def update_name(name, mapping, regex):
    m = regex.search(name)
    if m:
        street_type = m.group()
        if street_type in mapping:
            name = re.sub(regex, mapping[street_type], name)

    return name

for street_type, ways in DC_street_types.iteritems():
    for name in ways:
        new_name = update_name(name, mapping, street_re)
        print name, "=>", new_name

# Zip code
# audit zipcode
def audit_zipcode(invalid_zipcodes, zipcode):
    Digits = re.match(r'^\d{5}$', zipcode)
   
    if not Digits:
        invalid_zipcodes[Digits].add(zipcode)
    
    elif Digits:
        if len(zipcode) != 5:
            invalid_zipcodes[Digits].add(zipcode)
        
def is_zipcode(elem):
    return (elem.attrib['k'] == "addr:postcode")

def audit_zip(osmfile):
    osm_file = open(osmfile, "r")
    invalid_zipcodes = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_zipcode(tag):
                    audit_zipcode(invalid_zipcodes,tag.attrib['v'])

    return invalid_zipcodes

DC_zipcode = audit_zip(DC_OSM)
pprint.pprint(dict(DC_zipcode))

len(dict(DC_zipcode))


# updata zipcode format
def update_zipcode(zipcode):
    num = re.findall(r'^(\d{5})-\d{4}$', zipcode)
    if num:
        num = num[0]
        return num
for street_type, ways in DC_zipcode.iteritems():
    for name in ways:
        new_name = update_zipcode(name)
        print name, "=>", new_name

# Audit phone number
def audit_phone_number(invalid_phone_numbers, phone_number):
    Digits = re.match(r'^\d{10}$', phone_number)
    if not Digits:
        invalid_phone_numbers[Digits].add(phone_number)
    
    elif Digits:
        if len(phone_number) != 10:
            invalid_phone_numbers[Digits].add(phone_number)
        
def is_phone_number(elem):
    return (elem.attrib['k'] == "phone")

def audit_phone(osmfile):
    osm_file = open(osmfile, "r")
    invalid_phone_numbers = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_phone_number(tag):
                    audit_phone_number(invalid_phone_numbers,tag.attrib['v'])

    return invalid_phone_numbers

DC_phone = audit_phone(DC_OSM)
pprint.pprint(dict(DC_phone))

# Remove all "+1", "-", "." " ","()" format phone numbers
def update_phone(phone_number):
    phone_number = phone_number.translate(None, ' ()-.')
    if phone_number.startswith("+1"):
        phone_number = phone_number[2:]
    return phone_number
for street_type, ways in DC_phone.iteritems():
    for name in ways:
        new_name = update_phone(name)
        print name, "=>", new_name


#Convert XML to JSON
CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

def shape_element(e):
    node = {} 
    address = {}
    node['created'] = {}
    node['pos'] = [0,0]
    if e.tag == "way":
        node['node_refs'] = []
    if e.tag == "node" or e.tag == "way" :
        node['type'] = e.tag
        #attributes
        for k, v in e.attrib.iteritems():
            #latitude
            if k == 'lat':
                try:
                    lat = float(v)
                    node['pos'][0] = lat
                except ValueError:
                    pass
            #longitude
            elif k == 'lon':
                try:
                    lon = float(v)
                    node['pos'][1] = lon
                except ValueError:
                    pass
            #creation metadata
            elif k in CREATED:
                node['created'][k] = v
            else:
                node[k] = v
        #children
        for tag in e.iter('tag'):
            k = tag.attrib['k']
            v = tag.attrib['v']
            if problemchars.match(k):
                continue
            elif lower_colon.match(k):
                k_split = k.split(':')
                #address fields
                if k_split[0] == 'addr':
                    k_item = k_split[1]
                    #streets
                    if k_item == 'street':
                        v = update_name(v, mapping, street_re)                    
                    #postal codes
                    if k_item == 'postcode':
                        v = update_zipcode(v)
                    address[k_item] = v
                    continue
            else:                
                #phone numbers
                if(is_phone_number(tag)):
                    v = update_phone(v)
            node[k] = v
 
        if address:
            node['address'] = address                          
                      
        #way children
        if e.tag == "way":
            for n in e.iter('nd'):
                ref = n.attrib['ref']
                node['node_refs'].append(ref);
        return node
    else:
        return None
                                                                                                                                            
from bson import json_util

def process_map(file_in, pretty = False):  
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data
process_map(DC_OSM)  

# Data Overview
import signal
import subprocess
from pymongo import MongoClient

#create database
pro = subprocess.Popen('mongod', preexec_fn = os.setsid)
db_name = 'streetmap'

client = MongoClient('localhost:27017')
db = client[db_name]

#mongoimport command 
collection = DC_OSM[:DC_OSM.find('.')]
json_file = DC_OSM + '.json'

mongoimport = 'mongoimport -h 127.0.0.1:27017 ' +                   '--db ' + db_name +                   ' --collection ' + collection +                   ' --file ' + json_file

if collection in db.collection_names():
    print 'Dropping collection: ' + collection
    db[collection].drop() 

# Execute the command
print 'Executing: ' + mongoimport
subprocess.call(mongoimport.split())

# After importing, get the collection from the database.
dc = db[collection]

# Look at the size of the new JSON file, comparing to old downloaded one:
print "The DC.osm file size is {} mb.".format(os.path.getsize(DC_OSM) / 1.0e6)
print "The transformed JSON file size is {} mb.".format(os.path.getsize(DC_OSM + ".json") / 1.0e6)

# Number of Documents:
dc.find().count()

# Number of unique users:
len(dc.distinct('created.user'))

# Number of Nodes and Ways:
print "Number of nodes:",dc.find({'type':'node'}).count()
print "Number of ways:",dc.find({'type':'way'}).count()

# Name of top 10 contributors:
result = dc.aggregate([{"$group" : {"_id" : "$created.user", "count" : {"$sum" : 1}}}, 
                               {"$sort" : {"count" : - 1}}, {"$limit" : 10}])
for r in list(result):
    print(r)

# The top 5 Most Referenced Nodes:
top_5 = dc.aggregate([{'$unwind': '$node_refs'}, {'$group': {'_id': '$node_refs', 'count': {'$sum': 1}}}, 
                              {'$sort': {'count': -1}}, {'$limit': 5}])
pprint.pprint(top_5)  
print

for node in top_5:  
    pprint.pprint(dc.find({'id': node['_id']})[0])

# Number of Documents with Street Addresses:
dc.find({'address.street' : {'$exists' : 1}}).count()

# Number of Documents with postcode:
dc.find({'address.postcode' : {'$exists' : 1}}).count()

# Number of Documents with postcode:
dc.find({'phone' : {'$exists' : 1}}).count()

# List of top 10 cuisine in DC:
top_10 = dc.aggregate([{"$match":{"amenity":{"$exists":1},"amenity":"restaurant",}},      
                      {"$group":{"_id":{"Food":"$cuisine"},"count":{"$sum":1}}},
                      {"$project" : {"_id":0,"Food":"$_id.Food","Count":"$count"}},
                      {"$sort" : {"Count":-1}}, 
                      {"$limit": 10}])
for t in list(top_10):
    print(t)

# List of top 15 amenities in DC:
top_10 = dc.aggregate([{'$match': {'amenity': {'$exists': 1}}}, 
                      {'$group': {'_id': '$amenity', 'count': {'$sum': 1}}}, 
                      {'$sort': {'count': -1}}, 
                      {'$limit': 15}])

for t in list(top_10):
    print(t)

# List of top 10 Banks:
bank = dc.aggregate([{'$match': {'amenity': 'bank'}}, 
                      {'$group': {'_id': '$name', 'count': {'$sum': 1}}}, 
                      {'$sort': {'count': -1}},
                      {'$limit': 10}])
for b in list(bank):
    print(b)

# List of top 10 restaurants:
restaurant = dc.aggregate([{'$match': {'amenity': 'restaurant'}}, 
                      {'$group': {'_id': '$name', 'count': {'$sum': 1}}}, 
                      {'$sort': {'count': -1}},
                      {'$limit': 10}])

for r in list(restaurant):
    print(r)





