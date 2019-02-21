#!/usr/bin/env python
"""
Module is for importing and loading climate division data by both region and
state.
"""



def importStates():

	"""
	Creates a dictionary called 'states' where keys are the state code.
	Combined with importDivs, all possible combinations of state-div
	can be realized for loading the climdiv dataset (i.e. 0101 is Alabama
	division 1, etc)
	"""

	states = {
		'01' : 'Alabama',
		'02' : 'Arizona',
		'03' : 'Arkansas',
		'04' : 'California',
		'05' : 'Colorado',
		'06' : 'Connecticut',
		'07' : 'Delaware',
		'08' : 'Florida',
		'09' : 'Georgia',
		'10' : 'Idaho',
		'11' : 'Illinois',
		'12' : 'Indiana',
		'13' : 'Iowa',
		'14' : 'Kansas',
		'15' : 'Kentucky',
		'16' : 'Louisiana',
		'17' : 'Maine',
		'18' : 'Maryland',
		'19' : 'Massachusetts',
		'20' : 'Michigan',
		'21' : 'Minnesota',
		'22' : 'Mississippi',
		'23' : 'Missouri',
		'24' : 'Montana',
		'24' : 'Montana',
		'25' : 'Nebraska',
		'26' : 'Nevada',
		'27' : 'New Hampshire',
		'28' : 'New Jersey',
		'29' : 'New Mexico',
		'30' : 'New York',
		'31' : 'North Carolina',
		'32' : 'North Dakota',
		'33' : 'Ohio',
		'34' : 'Oklahoma',
		'35' : 'Oregon',
		'36' : 'Pennsylvania',
		'37' : 'Rhode Island',
		'38' : 'South Carolina',
		'39' : 'South Dakota',
		'40' : 'Tennessee',
		'41' : 'Texas',
		'42' : 'Utah',
		'43' : 'Vermont',
		'44' : 'Virginia',
		'45' : 'Washington',
		'46' : 'West Virginia',
		'47' : 'Wisconsin',
		'48' : 'Wyoming',
		}
	return states

def importDivs():

	"""
	Creates a dictionary with all the possible division numbers to match
	with state codes
	"""

	divnums = [
		'01',
		'02',
		'03',
		'04',
		'05',
		'06',
		'07',
		'08',
		'09',
		'10',
		'11',
		'12',
		'13'
		]
	return divnums

def importRegions():

	"""
	Creates a dictionary where region names are the keys, matched to state
	lists. For looping through regions to create plots, etc.
	"""

	regions = {
		'Northwest' : ['Washington','Oregon','Idaho'],
		'West'		: ['California', 'Nevada'],
		'Southwest'	: ['Utah','Arizona','New Mexico', 'Colorado'],
		'West North Central'	: ['Montana', 'Wyoming', 'Nebraska', \
									'North Dakota', 'South Dakota'],
		'South'	: ['Kansas','Oklahoma', 'Texas', 'Arkansas', 'Louisiana', \
					'Mississippi'],
		'East North Central'	: ['Minnesota', 'Iowa', 'Wisconsin', \
									'Michigan'],
		'Central'	: ['Missouri', 'Illinois', 'Indiana', 'Tennessee', \
						'Kentucky', 'West Virginia', 'Ohio'],
		'Southeast'	: ['Alabama', 'Georgia', 'Florida', 'South Carolina', \
						'North Carolina', 'Virginia'],
		'Northeast'	: ['Pennsylvania', 'Maryland', 'Delaware', 'New Jersey', \
						'Connecticut', 'Rhode Island', 'Massachusetts', \
						'New York', 'Vermont', 'New Hampshire', 'Maine']
				}

	return regions

def reverseStates(states):

	"""
	Enter in the states dictionary, and it reverses so that states are keys
	and state codes is matched to the state.
	"""

	codestates = {}
	for code in states:
		codestates[states[code]] = code
	return codestates

def importPhasecodes():
	phasecodes = {'allyears' : '01',
				'lanina':	'02',
				'neutneg':	'03',
				'neutral':	'04',
				'neutpos':	'05',
				'elnino':	'06'}
	return phasecodes
