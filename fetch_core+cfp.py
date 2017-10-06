#!/usr/bin/python3

import sys
import json
import requests
from datetime import datetime, date
from urllib.parse import urlparse
from bs4 import BeautifulSoup


def head(n, iterator):
	""" Return the first (up to) n elements of an iterator
	"""
	for _ in range(n):
		yield next(iterator)


def uniq(it, **sorted_kwargs):
	""" Sort the iterator using sorted(it, **sorted_kwargs) and return
		all non-duplicated elements (for !=)
	"""
	y = next(it)
	yield y
	for x in sorted(it, **sorted_kwargs):
		if x != y:
			yield x
			y = x


def get_soup(url, filename, **kwargs):
	""" Simple caching mechanism. Fetch a page from url and save it in filename.

		If filename exists, return its contents instead.
	"""
	try:
		with open(filename, 'r') as fh:
			soup = BeautifulSoup(fh.read(), 'lxml')
	except FileNotFoundError:
		r = requests.get(url, **kwargs)
		with open(filename, 'w') as fh:
			print(r.text, file=fh)
		soup = BeautifulSoup(r.text, 'lxml')
	return soup


ranks = ['A*', 'A', 'B', 'C', 'D']
def ranksort(conf): # lower is better
	""" Utility to sort the ranks based on the order we want (specificially A* < A).
	"""
	try: return (ranks.index(conf['rank']), conf['acronym'].lower())
	except ValueError: return (len(ranks), conf['acronym'].lower()) # non-ranked, e.g. 'Australasian'


def parse_date(d):
	return datetime.strptime(d, '%b %d, %Y').date()


def json_encode_dates(obj):
	if isinstance(obj, date):
		return str(obj)
	else:
		raise TypeError('{} not encodable'.format(obj))


url_corerank = 'http://portal.core.edu.au/conf-ranks/?search=&by=all&source=CORE2017&sort=arank&page={}'
url_forcodes = 'http://www.uq.edu.au/research/research-management/era-for-codes'
wiki_cfp = 'http://www.wikicfp.com'
url_cfpsearch = wiki_cfp + '/cfp/servlet/tool.search'

missing_values = {'N/A', 'TBD'}

hardcoded = { # Manually correct some errors. TODO this is not scalable.
	("SENSYS", 2017): ["03-04-2017", "10-04-2017", "17-07-2017", None, "05-11-2017", "08-11-2017"],
	("ISCA",   2018): ["14-11-2017", "21-11-2017", "13-03-2018", None, "02-06-2018", "06-06-2018"]
}

def get_forcodes():
	""" Fetch and return the mapping of For Of Research (FOR) codes to the corresponding names.
	"""
	forcodes = {}
	soup = get_soup(url_forcodes, 'cache/for_codes.html')
	for row in soup.find('table').findAll('tr'):
		try:
			code, field = [td.text.strip() for td in row.findAll('td')]
			if field:
				forcodes[code] = field.capitalize()
		except ValueError:
			print(row.text)
			raise

	return forcodes


def get_confs():
	""" Generator of all conferences listed on the core site, as dicts
	"""
	for p in range(32): #NB hardcoded number of pages on the core site.
		f = 'cache/ranked_{}-{}.html'.format(50 * p + 1, 50 * (p + 1))
		soup = get_soup(url_corerank.format(p), f)

		table = soup.find('table')
		rows = iter(table.findAll('tr'))

		headers = [r.text.strip().lower() for r in next(rows).findAll('th')]
		for row in rows:
			yield {h:v for h, v in zip(headers, [r.text.strip() for r in row.findAll('td')])}


def strip_trailing_paren(string):
	""" If string ends with a parenthesized part, remove it, e.g. "foo (bar)" -> "foo"
	"""
	return string[:string.index(' (')] if ' (' in string and string.rstrip()[-1] == ')' else string


def fetch_core():
	""" Generator returning all the conferences listed on the core site, sorted and unique,
		with field (for) translated to text, pretty dict keys, etc.
	"""
	field_codes = get_forcodes()

	for conf in uniq((c for c in get_confs() if ranksort(c)[0] < 4), key = ranksort):
		pretty_conf = {'Acronym': strip_trailing_paren(conf['acronym']),
				'Title': strip_trailing_paren(conf['title']),
				'CORE 2017 Rank': conf['rank']}

		if conf['for']:
			pretty_conf['Field'] = field_codes[conf['for']]

		yield pretty_conf


def parse_cfp_soup(soup):
	""" Parse a page from wiki-cfp. Return all useful data about the conference.
	"""
	# Find the the table containing the interesting data about the conference
	# There's always a "When" and a "Where" in the info table, even though they might be N/A
	for info_table in soup.find('th', text = 'Where').parents:
		if info_table.name == 'table':
			break
	else:
		raise ValueError('Cound not find parent table!')

	# Populate data with {left cell: right cell} for every line in the table
	it = ((tr.find('th').text, tr.find('td').text.strip()) for tr in info_table.findAll('tr'))
	data = {th: td for th, td in it if td not in missing_values}

	# find all links next to a "Link: " text, and return both their text and href values
	l = [t.parent.find('a', href = True) for t in soup.findAll(text = lambda t: 'Link: ' in t)]
	links = {link.text for link in l if link} | {link['href'] for link in l if link}

	if len(links) == 1:
		data['Link'] = links.pop()
	elif len(links) > 1:
		raise ValueError("ERROR Multiple distinct link values: " + ', '.join(links))

	return data


def find_link(search_soup, conf, year):
	""" Find the link to the conference page in the search page's soup
	"""
	search = '{} {}'.format(conf['Acronym'], year).lower()
	conf_link = search_soup.findAll('a', href = True, text = lambda t: t and t.lower() == search)
	print(search,': ',conf_link)

	for cl in conf_link:
		for tr in cl.parents:
			if tr.name == 'tr':
				break
		else:
			raise ValueError('Cound not find parent row!')
		conf_name = [td.text for td in tr.findAll('td') if td not in cl.parents]
		print(conf_name)
		print(conf['Title'])

		return wiki_cfp + cl['href']


def get_cfp(conf, year):
	""" Fetch the cfp from wiki-cfp for the given conference at the given year.
	"""
	f='cache/cfp_{}-{}.html'.format(conf['Acronym'].replace('/', '_'), year)
	try:
		with open(f, 'r') as fh:
			soup = BeautifulSoup(fh.read(), 'lxml')

	except FileNotFoundError:
		search_f='cache/search_cfp_{}-{}.html'.format(conf['Acronym'].replace('/', '_'), year)
		search_soup = get_soup(url_cfpsearch, search_f, params = {'q': conf['Acronym'], 'y': year})

		conf_page = find_link(search_soup, conf, year)
		if conf_page:
			soup = get_soup(conf_page, f)
		else:
			return {}

	try:
		return parse_cfp_soup(soup)
	except ValueError as e:
		print('Error on conference {}: {}'.format(conf, e), file=sys.stderr)
		raise


def update_confs(out):
	""" List all conferences from CORE, fetch their CfPs and print the output data as json to out.
	"""
	today = datetime.now().date()
	years = [today.year, today.year + 1]
	date_names = ['Abstract Registration Due', 'Submission Deadline', 'Notification Due', 'Final Version Due', 'Start', 'End']
	date_orig = ['orig-abstract', 'orig-submission', 'orig-notif', 'orig-camera', 'orig-start', 'orig-end']

	columns = ['Acronym', 'Title', 'CORE 2017 Rank'] + date_names + ['Field', 'Link'] + date_orig

	# Columns for which we output '(missing)' instead of None
	mark_missing = set(columns) - set(date_names) - set(date_orig)


	print('{"columns":', file=out);
	json.dump([{'title': c} for c in columns], out)
	print(',\n"data": [', file=out)
	writing_first_conf = True

	for conf in head(5, fetch_core()):
		last_year = {}
		for y in years:
			if (conf['Acronym'].upper(), y) in hardcoded:

				data = {dn: datetime.strptime(v, '%d-%m-%Y').date() for v, dn
							in zip(hardcoded[(conf['Acronym'].upper(), y)], date_names) if v}

				data.update({orig: True for v, orig
						in zip(hardcoded[(conf['Acronym'].upper(), y)], date_orig) if v})

			else:
				data = get_cfp(conf, y)

				if 'When' in data:
					data['Start'], data['End'] = data.pop('When').split(' - ')

				for dn, orig in zip(date_names, date_orig):
					if dn in data:
						data[orig] = True
						data[dn] = parse_date(data[dn])
					else:
						# add extrapolated date from previous year
						if dn in last_year:
							data[orig] = False
							data[dn] = last_year[dn].replace(year = last_year[dn].year + 1)
						else:
							continue
			if data:
				last_year = data
				data.update(conf)

				if max(data.get(c, today) for c in date_names) > today:
					if not writing_first_conf: print(',', file=out)
					else: writing_first_conf = False

					# filter out empty values for non-date columns
					json.dump([data.get(col, '(missing)') if col in mark_missing else data.get(col, None)
									for col in columns], out, default = json_encode_dates)

					break

	print('\n]}', file=out)


if __name__ == '__main__':
	with open('cfp.json', 'w') as out:
		update_confs(out)


