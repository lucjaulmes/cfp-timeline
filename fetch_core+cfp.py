#!/usr/bin/python3

import re
import sys
import json
import requests
import datetime
from requests.exceptions import ConnectionError, MissingSchema
from functools import total_ordering
from collections import defaultdict
from urllib.parse import urlparse
from bs4 import BeautifulSoup


def head(n, iterable):
	""" Generator listing the first (up to) n elements of an iterable

	Args:
		n (`int`): the maximum amount of elements to list
		iterable `iterable`: An iterable whose first elements we want to get
	"""
	_it = iter(iterable)
	for _ in range(n):
		yield next(_it)


def uniq(iterable, **sorted_kwargs):
	""" Sort the iterator using sorted(it, **sorted_kwargs) and return
	all non-duplicated elements.

	Args:
		iterable (iterable): the elements to be listed uniquely in order
		sorted_kwargs (`dict`): the arguments to be passed to sorted(iterable, ...)
	"""
	_it = iter(sorted(iterable, **sorted_kwargs))
	y = next(_it)
	yield y
	while True:
		x = next(_it)
		if x != y: yield x
		y = x


class PeekIter(object):
	""" Iterator that allows

	Attributes:
		_it (`iterable`): wrapped by this iterator
		_ahead (`list`): stack of the next elements to be returned by __next__
	"""
	_it = None
	_ahead = []

	def __init__(self, iterable):
		super(PeekIter, self)
		self._it = iterable
		self._ahead = []


	def __iter__(self):
		return self


	def __next__(self):
		if self._ahead:
			return self._ahead.pop(0)
		else:
			return next(self._it)


	def peek(self, n = 0):
		""" Tries peeking ahead. Raises IndexError if we go beyond the iterator length.

		Args:
			n (`int`): how many positions ahead to look in the iterator, 0 means next element.
		"""
		if n < 0: raise ValueError('n < 0 but can not peek back, only ahead')

		try:
			self._ahead.extend(next(self._it) for _ in range(n - len(self._ahead) + 1))
		except StopIteration:
			raise IndexError

		return self._ahead[-n - 1]


def memoize(f):
	""" A decorator that replaces a function f with a wrapper caching its result.
	The cached result is computed only at the first call, and stored in an attribute of f.

	Args:
		f (`function`): A function whose output needs to be (lazily) cached

	Returns:
		`function`: The wrapper that calls f, caches its result, and serves it
	"""
	def wrapper(*args, **kwargs):
		if not hasattr(f, '_cached'): f._cached = f(*args, **kwargs)
		return f._cached

	return wrapper


def get_soup(url, filename, **kwargs):
	""" Simple caching mechanism. Fetch a page from url and save it in filename.

	If filename exists, return its contents instead.
	"""
	try:
		with open(filename, 'r') as fh:
			soup = BeautifulSoup(fh.read(), 'lxml')
	except FileNotFoundError:
		if url:raise ConnectionError # DEBUG
		r = requests.get(url, **kwargs)
		with open(filename, 'w') as fh:
			print(r.text, file=fh)
		soup = BeautifulSoup(r.text, 'lxml')
	return soup


class ConfMetaData(object):
	""" Heuristic to reduce a conference title to a matchable set of words.

	Args:
		title (`str`): the full title or string describing the conference (containing the title)
		acronym (``): the acronym or short name of the conference
		year (`int` or `str`): the year of the conference
	"""

	#ACM Special Interest Groups
	_sig = set(map(str.lower, {"ACCESS", "ACT", "Ada", "AI", "APP", "ARCH", "BED", "Bio", "CAS", "CHI", "COMM", "CSE", "DA",
			"DOC", "ecom", "EVO", "GRAPH", "HPC", "IR", "ITE", "KDD", "LOG", "METRICS", "MICRO", "MIS", "MM", "MOBILE",
			"MOD", "OPS", "PLAN", "SAC", "SAM", "SIM", "SOFT", "SPATIAL", "UCCS", "WEB"}))

	_org = {'acm', 'ieee', 'ifip', 'siam', 'usenix'} | {"sig"+s for s in _sig}

	_tens = {'twenty', 'thirty', 'fourty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'}
	_ordinal = re.compile(r'[0-9]+(st|nd|rd|th)|(({tens})?(first|second|third|(four|fif|six|seven|eigh|nine?)th))|(ten|eleven|twelf|(thir|fourt|fif|six|seven|eigh|nine?)teen)th'.format(tens = '|'.join(_tens)))

	topic_keywords = None
	organisers = None
	number = None
	type_ = None
	qualifiers = None


	def __init__(self, title, acronym, year = '', **kwargs):
		super(ConfMetaData, self).__init__(**kwargs)

		self.topic_keywords = []
		self.organisers = set()
		self.number = ''
		self.type_ = ''
		self.qualifiers = []

		if acronym.startswith('sig') and acronym[3:] in self._sig:
			# temporary, want to see if this happens. E.g. SIGMETRICS?
			raise ValueError('Special case acronym == interest group: {}'.format(acronym))

		# lower case, replace characters in dict by whitepace, repeated spaces will be removed by split()
		words = title.lower().translate({ord(c):' ' for c in "-/&,():_~'"}).split()

		# remove articles, conjunctions, acronym and year of conference if they are repeated
		# semantically filter conference editors/organisations, special interest groups (sig...), etc.
		words = PeekIter(w for w in words if w not in {'the', 'on', 'for', 'of', 'in', 'and', 'cfp', acronym.lower(), str(year), str(year)[2:]})

		for w in words:
			if w == "sig":
				try:
					if words.peek() in self._sig:
						self.organisers.add(w + next(words))
						continue
				except IndexError: pass

			if w in self._tens:
				try:
					m = self._ordinal.match(words.peek())
					if m:
						self.number = w + '-' + m.group(0)
						next(words)
						continue
				except IndexError: pass


			if w in self._org:
				self.organisers.add(w)

			elif w in {'congress', 'conference', 'seminar', 'symposium', 'workshop', 'tutorial'}:
				self.type_ += w

			elif w in {'annual', 'european', 'international', 'joint', 'national'}:
				self.qualifiers.append(w)

			else:
				m = ConfMetaData._ordinal.match(w)
				if m:
					self.number = m.group(0)
					continue

				self.topic_keywords.append(w)
		# TODO Expand acronyms such as OS A/V, etc: Network Os Support Digital A V == Network Operating Systems Support Digital Audio Video
		# Also in the other way: Special Interest Group [...] -> sig... , Call for papers: cfp, etc.


	def topic(self, sep = ' '):
		return sep.join(self.topic_keywords).title()


	@staticmethod
	def _set_diff(left, right):
		""" Return an int quantifying the difference between the sets.

		Penalize a bit for difference on a single side, more for differences on both sides, under the assumption that
		partial information on one side is better than dissymetric information
		"""
		n_comm = len(set(left) & set(right))
		# for 4 diffs => 4 + 0 -> 5, 3 + 1 -> 8, 2 + 2 -> 9
		try:
			return ((1 + len(left) - n_comm) * (1 + len(right) - n_comm) - 1) / (len(left) + len(right) - n_comm)
		except ZeroDivisionError:
			return 0


	@staticmethod
	def _list_diff(left, right):
		""" Return an int quantifying the difference between the sets

		Uset the same as `~set_diff` and add penalties for dfferences in word order.
		"""
		# for 4 diffs => 4 + 0 -> 5, 3 + 1 -> 8, 2 + 2 -> 9
		comm = set(left) & set(right)
		try:
			set_diff = ((1 + len(left) - len(comm)) * (1 + len(right) - len(comm)) - 1) / (len(left) + len(right) - len(comm))
			sort_diff = sum(abs(left.index(c) - right.index(c)) for c in comm) / (len(left) + len(right))

			return set_diff + sort_diff
		except ZeroDivisionError:
			return 0


	@staticmethod
	def _difference(self, other):
		""" Compare the two ConfMetaData instances and rate how similar they are.
		"""
		return (self._set_diff({self.type_}, {other.type_}),
				self._set_diff(self.organisers, other.organisers),
				self._list_diff(self.topic_keywords, other.topic_keywords),
				self._list_diff(self.qualifiers, other.qualifiers),
				self._set_diff({self.number}, {other.number})
		)


@total_ordering
class Conference(ConfMetaData):
	__slots__ = ('acronym', 'title', 'rank', 'field')
	_ranks = ['A*', 'A', 'B', 'C', 'D', 'E']

	def __init__(self, title, acronym, rank = None, field = None, **kwargs):
		super(Conference, self).__init__(title, acronym, **kwargs)

		self.title = title
		self.acronym = acronym
		self.rank = rank or '(missing)'
		self.field = field or '(missing)'


	def ranksort(self): # lower is better
		""" Utility to sort the ranks based on the order we want (specificially A* < A).
		"""
		try: return self._ranks.index(self.rank)
		except ValueError: return len(self._ranks) # non-ranked, e.g. 'Australasian'


	def __eq__(self, other):
		return (self.rank, self.acronym, self.title, other.field) == (other.rank, other.acronym, other.title, other.field)


	def __lt__(self, other):
		return (self.ranksort(), self.acronym, self.title, other.field) < (other.ranksort(), other.acronym, other.title, other.field)



class CallForPapers(ConfMetaData):
	_base_url = None
	_url_cfpsearch = None
	_url_cfpseries = None

	_date_fields = ['abstract', 'submission', 'notification', 'camera_ready', 'conf_start', 'conf_end']
	_date_names = ['Abstract Registration Due', 'Submission Deadline', 'Notification Due', 'Final Version Due', 'Start', 'End']

	__slots__ = ('conf', 'desc', 'dates', 'orig', 'url_cfp', 'year', 'link')


	def __init__(self, conf, desc, year = '', url_cfp = None, **kwargs):
		# Initialize parent parsing with the description
		super(CallForPapers, self).__init__(desc, conf.acronym, year, **kwargs)

		self.conf = conf
		self.desc = desc
		self.year = year
		self.dates = {}
		self.orig = {}
		self.link = '(missing)'
		self.url_cfp = url_cfp


	def extrapolate_missing_dates(self, prev_year_cfp):
		# NB: it isn't always year = this.year, e.g. the submission can be the year before the conference dates
		# TODO: smarter extrapolation using delays instead of dates?
		for field in (field for field in prev_year_cfp.dates if field not in self.dates):
			n = self._date_fields.index(field)
			self.dates[field] = prev_year_cfp.dates[field].replace(year = prev_year_cfp.dates[field].year + 1)
			self.orig[field] = False


	@classmethod
	def parse_confseries(cls, soup):
		raise NotImplementedError


	@classmethod
	def parse_search(cls, conf, year, soup):
		raise NotImplementedError


	def parse_cfp(self, soup):
		raise NotImplementedError


	@classmethod
	@memoize
	def get_conf_series(cls):
		""" Returns map of all conference series listed on the core site, as dicts: acronym -> list of (conf name, link) tuples
		"""
		conf_series = defaultdict(lambda: [])
		for i in (chr(ord('A') + x) for x in range(26)):
			f='cache/cfp_series_{}.html'.format(i)
			soup = get_soup(cls._url_cfpseries.format(initial = i), f)

			for acronym, name, link in cls.parse_confseries(soup):
				conf_series[acronym].append((ConfMetaData(title = name, acronym = acronym), link))

		return dict(conf_series)


	def fetch_cfp_data(self):
		""" Parse a page from wiki-cfp. Return all useful data about the conference.
		"""
		f = 'cache/' + 'cfp_{}-{}_{}.html'.format(self.conf.acronym, self.year, self.conf.topic()).replace('/', '_') # topic('-')
		self.parse_cfp(get_soup(self.url_cfp, f))


	@classmethod
	def find_link(cls, conf, year):
		""" Find the link to the conference page in the search page's soup
		"""
		search_f = 'cache/' + 'search_cfp_{}-{}.html'.format(conf.acronym, year).replace('/', '_')
		soup = get_soup(cls._url_cfpsearch, search_f, params = {'q': conf.acronym, 'y': year})

		# DEBUG
		#options = (CallForPapers(conf, desc, year, url) for desc, url in cls.parse_search(conf, year, soup))
		#return min(options, key = CallForPapers.rating)

		options = sorted((cls(conf, desc, year, url) for desc, url in cls.parse_search(conf, year, soup)), key = lambda o: sum(o.rating()))
		print('Searching for {} ({}) {}#{}#{}'.format(conf.acronym, conf.rank, year, conf.title, conf.topic()))
		if options:
			for o in options:
				ratings = map('{:.2f}'.format, o.rating())
				print('typ,org,tpc,qlf={}#{}#{}'.format(' '.join(ratings), o.desc, o.topic()))
			return options[0]
		else:
			raise ValueError


	@classmethod
	def get_cfp(cls, conf, year):
		""" Fetch the cfp from wiki-cfp for the given conference at the given year.
		"""
		try:
			raise MissingSchema # DEBUG
			# Try our cache without knowing the URL, will raise MissingSchema if we  miss in cache and try to access it
			cfp = cls(conf, desc = '', year = year)
			cfp.fetch_cfp_data()
			return cfp

		except MissingSchema: pass

		try:
			cfp = cls.find_link(conf, year)
			if cfp:
				cfp.fetch_cfp_data()
				return cfp

		except ConnectionError: pass #print('Connection error when fetching search for {} {}'.format(conf.acronym, year))
		except ValueError: pass


	@classmethod
	def columns(cls):
		""" Return column titles for cfp data.
		"""
		return ['Acronym', 'Title', 'CORE 2017 Rank'] + cls._date_names + ['Field', 'Link'] + ['orig_' + d for d in cls._date_fields]


	def values(self):
		""" Return values of cfp data, in column order.
		"""
		return [self.conf.acronym, self.conf.title, self.conf.rank] + [self.dates.get(f, None) for f in self._date_fields] + \
										[self.conf.field, self.link] + [self.orig.get(f, None) for f in self._date_fields]


	def max_date(self):
		""" Get the max date in the cfp
		"""
		return max(self.dates.values())


	def rating(self):
		""" Rate the (in)adequacy of the cfp with its conference: lower is better.
		"""
		return self._difference(self, self.conf)[:4]


class WikicfpCFP(CallForPapers):
	_base_url = 'http://www.wikicfp.com'
	_url_cfpsearch = _base_url + '/cfp/servlet/tool.search'
	_url_cfpseries = _base_url + '/cfp/series?t=c&i={initial}'


	@staticmethod
	def parse_date(d):
		return datetime.datetime.strptime(d, '%b %d, %Y').date()


	@classmethod
	def parse_confseries(cls, soup):
		""" Given the BeautifulSoup of a CFP series list page, generate all (acronym, description, url) tuples for links that
		point to conference series.
		"""
		links = soup.findAll('a', {'href': lambda l:l.startswith('/cfp/program')})
		return (tuple(l.parent.text.strip().split(' - ', 1)) + (cls._base_url + ['href']) for l in links)


	@classmethod
	def parse_search(cls, conf, year, soup):
		""" Given the BeautifulSoup of a CFP search page, generate all (description, url) tuples for links that seem
		to correspond to the conference and year requested.
		"""
		search = '{} {}'.format(conf.acronym, year).lower()
		for conf_link in soup.findAll('a', href = True, text = lambda t: t and t.lower() == search):
			for tr in conf_link.parents:
				if tr.name == 'tr':
					break
			else:
				raise ValueError('Cound not find parent row!')

			# returns 2 td tags, one contains the link, the other the description
			conf_name = [td.text for td in tr.findAll('td') if td not in conf_link.parents]
			yield (conf_name[0], cls._base_url + conf_link['href'])


	def parse_cfp(self, soup):
		""" Given the BeautifulSoup of the CFP page, update self.dates and self.link
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
		data = {th: td for th, td in it if td not in {'N/A', 'TBD'}}

		if 'When' in data:
			data['Start'], data['End'] = data.pop('When').split(' - ')

		for f, name in zip(self._date_fields, ['Abstract Registration Due', 'Submission Deadline', 'Notification Due', 'Final Version Due', 'Start', 'End']):
			try:
				self.dates[f] = data[name] if isinstance(data[name], datetime.date) else self.parse_date(data[name])
				self.orig[f] = True
			except KeyError:
				pass # Missing date in data

		# find all links next to a "Link: " text, and return both their text and href values
		l = [t.parent.find('a', href = True) for t in soup.findAll(text = lambda t: 'Link: ' in t)]
		links = {link.text for link in l if link} | {link['href'] for link in l if link}

		if links:
			self.link = links.pop()

		if links:
			raise ValueError("ERROR Multiple distinct link values: " + ', '.join(links | {self.link}))


class CoreRanking(object):
	""" Utility class to scrape CORE conference listings and generate `~Conference` objects.
	"""
	_url_corerank = 'http://portal.core.edu.au/conf-ranks/?search=&by=all&source=CORE2017&sort=arank&page={}'
	_url_forcodes = 'http://www.uq.edu.au/research/research-management/era-for-codes'

	_historical = re.compile(r'\b(previous(ly)?|was|(from|pre) [0-9]{4}|merge[dr])\b', re.IGNORECASE)

	@classmethod
	@memoize
	def get_forcodes(cls):
		""" Fetch and return the mapping of For Of Research (FOR) codes to the corresponding names.
		"""
		forcodes = {}

		soup = get_soup(cls._url_forcodes, 'cache/for_codes.html')
		for row in soup.find('table').findAll('tr'):
			try:
				code, field = [td.text.strip() for td in row.findAll('td')]
				if field:
					forcodes[code] = field.title()
			except ValueError:
				raise

		return forcodes


	@classmethod
	def strip_trailing_paren(cls, string):
		""" If string ends with a parenthesized part, remove it, e.g. "foo (bar)" -> "foo"
		"""
		string = string.strip()
		try:
			paren = string.index(' (')
			if string[-1] == ')' and cls._historical.search(string[paren + 2:-1]):
				return string[:paren - 2]
		except ValueError:
			pass
		return string


	@classmethod
	def fetch_confs(cls):
		""" Generator of all conferences listed on the core site, as dicts
		"""
		with open('core.csv', 'w') as csv:
			print('title;acronym;rank;field', file=csv)

			for p in range(32): #NB hardcoded number of pages on the core site.
				f = 'cache/ranked_{}-{}.html'.format(50 * p + 1, 50 * (p + 1))
				soup = get_soup(cls._url_corerank.format(p), f)

				table = soup.find('table')
				rows = iter(table.findAll('tr'))

				headers = [r.text.strip().lower() for r in next(rows).findAll('th')]
				forcodes = cls.get_forcodes()

				tpos = headers.index('title')
				apos = headers.index('acronym')
				rpos = headers.index('rank')
				fpos = headers.index('for')

				for row in rows:
					val = [r.text.strip() for r in row.findAll('td')]
					yield Conference(cls.strip_trailing_paren(val[tpos]), val[apos], val[rpos], forcodes.get(val[fpos], None))

					print(';'.join(map(str, (cls.strip_trailing_paren(val[tpos]), val[apos], val[rpos], forcodes.get(val[fpos], None)))), file=csv)


def json_encode_dates(obj):
	if isinstance(obj, datetime.date):
		return str(obj)
	else:
		raise TypeError('{} not encodable'.format(obj))


def update_confs(out):
	""" List all conferences from CORE, fetch their CfPs and print the output data as json to out.
	"""
	today = datetime.datetime.now().date()
	years = [today.year, today.year + 1]

	hardcoded = { # Manually correct some errors. TODO this is not scalable.
		("SENSYS", 2017): ["03-04-2017", "10-04-2017", "17-07-2017", None, "05-11-2017", "08-11-2017"],
		("ISCA",   2018): ["14-11-2017", "21-11-2017", "13-03-2018", None, "02-06-2018", "06-06-2018"]
	}


	print('{"columns":', file=out);
	json.dump([{'title': c} for c in CallForPapers.columns()], out)
	print(',\n"data": [', file=out)
	writing_first_conf = True

	for conf in uniq(CoreRanking.fetch_confs()):
		if conf.ranksort() > 1: break # DEBUG

		last_year = None
		for y in years:
			override_dates = hardcoded.get((conf.acronym.upper(), y), None)
			if override_dates:
				cfp = CallForPapers(conf, desc = '', year = y)
				cfp.dates = {n: datetime.datetime.strptime(v, '%d-%m-%Y').date() for v, n in zip(override_dates, CallForPapers._date_names) if v}
				cfp.orig  = {n: True for v, n in zip(override_dates, CallForPapers._date_names) if v}

			else:
				cfp = WikicfpCFP.get_cfp(conf, y)
				# possibly try other CFP providers?

				if cfp and last_year:
					cfp.extrapolate_missing_dates(last_year)

			if cfp:
				last_year = cfp

				if cfp.max_date() > today:
					if not writing_first_conf: print(',', file=out)
					else: writing_first_conf = False

					# filter out empty values for non-date columns
					json.dump(cfp.values(), out, default = json_encode_dates)

					break

	print('\n]}', file=out)


if __name__ == '__main__':
	with open('cfp.json', 'w') as out:
		update_confs(out)


