#!/usr/bin/python3

import re
import os
import sys
import json
import glob
import shutil
import inflection
import requests
import datetime
from requests.exceptions import ConnectionError, MissingSchema
from functools import total_ordering
from collections import defaultdict
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qs, urlencode
from bs4 import BeautifulSoup

from time import time as get_time
from sys import stdout
from math import floor

try:
	from enchant import Dict
except ImportError:
	def Dict(*args):
		return None


class CFPNotFoundError(Exception):
	pass


class CFPCheckError(Exception):
	pass


class Progress():
	maxpos = 0.0
	next_pos = 0
	update_freq = 0
	start = 0
	template = "\rProgress{}: {{: 3.1f}} %\tElapsed: {{:2}}:{{:02}}\tETA: {{:2}}:{{:02}} "

	def change_max(self, maxpos):
		""" Change the number of items on hwihc me iterate.
		"""
		self.maxpos = float(maxpos)
		self.update_freq = max(1, int(floor(maxpos / 1000)))
		self.next_pos = 1

	def _print_update(self, pos):
		if pos < self.next_pos:
			return

		self.next_pos = pos + self.update_freq

		if pos < self.maxpos:
			ratio = pos / self.maxpos
			elapsed = get_time() - self.start
			est_remaining = elapsed * (1 - ratio) / ratio

			stdout.write(self.template.format(100 * ratio, *(divmod(int(elapsed), 60) + divmod(int(est_remaining), 60))))
			stdout.flush()

	def update_diff(self, pos):
		""" Wrap a pos != maxpos test to print updates
		"""
		self._print_update(pos)
		return pos != self.maxpos

	def update_less(self, pos):
		""" Wrap a pos < maxpos test to print updates
		"""
		self._print_update(pos)
		return pos < self.maxpos

	update = update_less


	def iterate(self, iterable, *enum_args):
		""" Wrap an iterable, yielding all its elements, to print updates
		"""
		if self.maxpos == 0.0:
			try:
				self.change_max(len(iterable))
			except TypeError as e:
				raise TypeError(("{}\nYou can use Progress.iterate() with an iterable that has no len() "
					+ "by providing its (expected) length to the constructor: Progress(maxpos = ...).\n").format(e.args[0]))


		for pos, item in enumerate(iterable, *enum_args):
			self._print_update(pos)
			yield item


	def clean_print(self, string, *print_args):
		""" Want to cleanly print a line in between progress updates? No problem!
		"""
		print('\r{}\r'.format(' ' * shutil.get_terminal_size().columns) + string, *print_args)
		self.next_pos = 1


	def __init__(self, maxpos = 0.0, operation = ""):
		self.template = self.template.format(' ' + operation if operation else '')
		self.change_max(maxpos)

	def __enter__(self):
		self.start = get_time()
		self.next_pos = self.update_freq
		stdout.write(self.template.split('\t')[0].format(0))
		stdout.flush()
		return self

	def __exit__(self, type, value, traceback):
		print(self.template.format(100, *(divmod(int(get_time() - self.start), 60) + (0, 0))))


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
		""" Returns next element(s) that will be returned from the iterator.

		Args:
			n (`int`): Number of positions to look ahead in the iterator.
					0 (by default) means next element, raises IndexError if there is none.
					Any value n > 0 returns a list of length up to n.
		"""
		if n < 0: raise ValueError('n < 0 but can not peek back, only ahead')

		try:
			self._ahead.extend(next(self._it) for _ in range(n - len(self._ahead) + 1))
		except StopIteration:
			pass

		if n == 0:
			return self._ahead[0]
		else:
			return self._ahead[:n]


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
		r = requests.get(url, **kwargs)
		with open(filename, 'w') as fh:
			print(r.text, file=fh)
		soup = BeautifulSoup(r.text, 'lxml')
	return soup


def normalize(string):
	# Asia -> Asium and Meta -> Metum, really?
	return inflection.singularize(string.lower()) if len(string) > 3 else string.lower()


class ConfMetaData(object):
	""" Heuristic to reduce a conference title to a matchable set of words.

	Args:
		title (`str`): the full title or string describing the conference (containing the title)
		acronym (``): the acronym or short name of the conference
		year (`int` or `str`): the year of the conference
	"""

	# associations, societies, institutes, etc. that organize conferences
	_org = {'ACIS':'Association for Computer and Information Science', 'ACL':'Association for Computational Linguistics', 'ACM':'Association for Computing Machinery',
			'ACS':'Arab Computer Society', 'AoM':'Academy of Management', 'CSI':'Computer Society of Iran', 'DIMACS':'Center for Discrete Mathematics and Theoretical Computer Science',
			'ERCIM':'European Research Consortium for Informatics and Mathematics', 'Eurographics':'Eurographics', 'Euromicro':'Euromicro',
			'IADIS':'International Association for the Development of the Information Society', 'IAPR':'International Association for Pattern Recognition',
			'IAVoSS':'International Association for Voting Systems Sciences', 'ICSC':'ICSC Interdisciplinary Research', 'IEEE':'Institute of Electrical and Electronics Engineers',
			'IFAC':'International Federation of Automatic Control', 'IFIP':'International Federation for Information Processing', 'IMA':'Institute of Mathematics and its Applications',
			'KES':'KES International', 'MSRI':'Mathematical Sciences Research Institute', 'RSJ':'Robotics Society of Japan', 'SCS':'Society for Modeling and Simulation International',
			'SIAM':'Society for Industrial and Applied Mathematics', 'SLKOIS':'State Key Laboratory of Information Security', 'SIGOPT':'DMV Special Interest Group in Optimization',
			'SIGNLL':'ACL Special Interest Group in Natural Language Learning', 'SPIE':'International Society for Optics and Photonics',
			'TC13':'IFIP Technical Committee on Human–Computer Interaction', 'Usenix':'Advanced Computing Systems Association', 'WIC':'Web Intelligence Consortium'}

	#ACM Special Interest Groups
	_sig = {'ACCESS':'Accessible Computing', 'ACT':'Algorithms Computation Theory', 'Ada':'Ada Programming Language', 'AI':'Artificial Intelligence',
			'APP':'Applied Computing', 'ARCH':'Computer Architecture', 'BED':'Embedded Systems', 'Bio':'Bioinformatics', 'CAS':'Computers Society',
			'CHI':'Computer-Human Interaction', 'COMM':'Data Communication', 'CSE':'Computer Science Education', 'DA':'Design Automation',
			'DOC':'Design Communication', 'ecom':'Electronic Commerce', 'EVO':'Genetic Evolutionary Computation', 'GRAPH':'Computer Graphics Interactive Techniques',
			'HPC':'High Performance Computing', 'IR':'Information Retrieval', 'ITE':'Information Technology Education', 'KDD':'Knowledge Discovery Data',
			'LOG':'Logic Computation', 'METRICS':'Measurement Evaluation', 'MICRO':'Microarchitecture', 'MIS':'Management Information Systems', 'MM':'Multimedia',
			'MOBILE':'Mobility Systems, Users, Data Computing', 'MOD':'Management Data', 'OPS':'Operating Systems', 'PLAN':'Programming Languages',
			'SAC':'Security, Audit Control', 'SAM':'Symbolic Algebraic Manipulation', 'SIM':'Simulation Modeling', 'SOFT':'Software Engineering',
			'SPATIAL':'SIGSPATIAL', 'UCCS':'University College Computing Services', 'WEB':'Hypertext Web', 'ART': 'Artificial Intelligence'} # NB ART was renamed AI

	_meeting_types = {'congress', 'conference', 'seminar', 'symposium', 'workshop', 'tutorial'}
	_qualifiers = {'american', 'asian', 'australasian', 'australian', 'annual', 'biennial', 'european', 'iberoamerican', 'international', 'joint', 'national'}
	_replace = { # remove shortenings and typos, and americanize text
			**{'intl': 'international', 'conf': 'conference', 'dev': 'development'},
			**{'visualisation':'visualization', 'modelling':'modeling', 'internationalisation':'internationalization', 'defence':'defense',
				'standardisation':'standardization', 'organisation':'organization', 'optimisation':'optimization,', 'realising':'realizing', 'centre':'center'},
			**{'syste':'system', 'computi':'computing', 'artifical':'artificial', 'librari':'library', 'databa':'database,', 'conferen':'conference',
				'bioinformatic':'bioinformatics', 'symposi':'symposium', 'evoluti':'evolution', 'proce':'processes', 'provi':'proving', 'techology':'technology',
				'bienniel':'biennial', 'entertainme':'entertainment', 'retriev':'retrieval', 'engineeri':'engineering', 'sigraph':'siggraph',
				'intelleligence':'intelligence', 'simululation':'simulation', 'inteligence':'intelligence', 'manageme':'management', 'applicatio':'application',
				'developme':'development', 'cyberworl':'cyberworld', 'scien':'science', 'personalizati':'personalization', 'computati':'computation',
				'implementati':'implementation', 'languag':'language', 'traini':'training', 'servic':'services', 'intenational':'international', 'complexi':'complexity',
				'storytelli':'storytelling', 'measureme':'measurement', 'comprehensi':'comprehension', 'synthe':'synthesis', 'evaluatin':'evaluation', 'technologi':'technology'}
			}

	# NB simple acronym management, only works while first word -> acronym mapping is unique
	_acronyms = {''.join(s[0] for s in a.split()):[normalize(s) for s in a.split()] for a in \
				{'call for papers', 'geographic information system', 'high performance computing', 'message passing interface', 'object oriented', 'operating system',
					'parallel virtual machine', 'public key infrastructure', 'special interest group'}}
	# Computer Performance Evaluation ? Online Analytical Processing: OLAP? aspect-oriented programming ?

	_tens = {'twenty', 'thirty', 'fourty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'}
	_ordinal = re.compile(r'[0-9]+(st|nd|rd|th)|(({tens})?(first|second|third|(four|fif|six|seven|eigh|nine?)th))|(ten|eleven|twelf|(thir|fourt|fif|six|seven|eigh|nine?)teen)th'.format(tens = '|'.join(_tens)))

	_sigcmp = {normalize('SIG' + s):s for s in _sig}
	_orgcmp = {normalize(s):s for s in _org}

	_acronym_start = {v[0]:a for a, v in _acronyms.items()}
	_sig_start = {normalize(v.split()[0]):a for a, v in _sig.items() if a != 'ART'}

	_dict = Dict('EN_US')
	_misspelled = {}

	topic_keywords = None
	organisers = None
	number = None
	type_ = None
	qualifiers = None


	def __init__(self, title, conf_acronym, year = '', **kwargs):
		super(ConfMetaData, self).__init__(**kwargs)

		self.topic_keywords = []
		self.organisers = set()
		self.number = set()
		self.type_ = set()
		self.qualifiers = []

		# lower case, replace characters in dict by whitepace, repeated spaces will be removed by split()
		words = PeekIter(normalize(w) for w in title.translate({ord(c):' ' for c in "-/&,():_~'."}).split() \
							if normalize(w) not in {'the', 'on', 'for', 'of', 'in', 'and', str(year)})

		# semantically filter conference editors/organisations, special interest groups (sig...), etc.
		for w in words:
			try:
				w = self._replace[w]
			except KeyError: pass

			if w in self._orgcmp:
				self.organisers.add(self._orgcmp[w])
				continue

			if w in self._meeting_types:
				self.type_.add(w)
				continue

			# Also seen but risk colliding with topic words: Mini Conference, Working Conference
			if w in self._qualifiers:
				self.qualifiers.append(w)
				continue

			# Recompose acronyms
			try:
				acronym = self._acronym_start[w]
				next_words = self._acronyms[acronym][1:]

				if words.peek(len(next_words)) == next_words:
					self.topic_keywords.append(normalize(acronym))
					for _ in next_words: next(words)
					continue

				# TODO some acronyms have special characters, e.g. A/V, which means they appear as 2 words
			except KeyError: pass

			# Some specific attention brought to ACM special interest groups
			if w.startswith('sig'):
				try:
					if len(w) > 3:
						self.organisers.add('SIG' + self._sigcmp[w])
						continue

					sig = normalize('SIG' + next(words))
					if sig in self._sigcmp:
						self.organisers.add(self._sigcmp[sig])
						continue

					elif words.peek() in self._sig_start:
						sig = self._sig_start[words.peek()]
						next_words = [normalize(s) for s in self._sig[sig].split()][1:]
						if next_words == words.peek(len(next_words)):
							self.organisers.add('SIG' + sig)
							for _ in next_words: next(words)
							continue

				except (KeyError, IndexError): pass

			# three-part management for ordinals, to handle joint/separate words: twenty-fourth, 10 th, etc.
			if w in self._tens:
				try:
					m = self._ordinal.match(words.peek())
					if m:
						self.number.add(w + '-' + m.group(0))
						next(words)
						continue
				except IndexError: pass


			if w.isnumeric():
				try:
					if words.peek() == inflection.ordinal(int(w)):
						self.number.add(w + next(words))
						continue
				except IndexError:pass

			m = ConfMetaData._ordinal.match(w)
			if m:
				self.number.add(m.group(0))
				continue

			# acronym and year of conference if they are repeated
			if w == normalize(conf_acronym):
				try:
					if words.peek() == str(year)[2:]: next(words)
				except IndexError: pass
				continue

			# anything surviving to this point surely describes the topic of the conference
			self.topic_keywords.append(w)
			if self._dict and not self._dict.check(w):
				if w not in (normalize(s) for s in self._dict.suggest(w)):
					self._misspelled[w] = (conf_acronym, title)


	def topic(self, sep = ' '):
		return sep.join(self.topic_keywords).title()


	@staticmethod
	def _set_diff(left, right):
		""" Return an int quantifying the difference between the sets. Lower is better.

		Penalize a bit for difference on a single side, more for differences on both sides, under the assumption that
		partial information on one side is better than dissymetric information
		"""
		n_common = len(set(left) & set(right))
		l = len(left) - n_common
		r = len(right) - n_common

		if len(left) > 0 and len(right) > 0 and n_common == 0:
			return 1000
		else:
			return  l + r + 10 * l * r - 2 * n_common


	@staticmethod
	def _list_diff(left, right):
		""" Return an int quantifying the difference between the sets

		Uset the same as `~set_diff` and add penalties for dfferences in word order.
		"""
		# for 4 diffs => 4 + 0 -> 5, 3 + 1 -> 8, 2 + 2 -> 9
		common = set(left) & set(right)
		n_common = len(common)
		l = [w for w in left if w in common]
		r = [w for w in right if w in common]
		n_l, n_r = len(l) - n_common, len(r) - n_common
		try:
			mid = round(sum(l.index(c) - r.index(c) for c in common) / len(common))
			sort_diff = sum(abs(l.index(c) - r.index(c) - mid) for c in common) / n_common
		except ZeroDivisionError:
			sort_diff = 0

		# disqualify if there is nothing in common
		if left and right and not common:
			return 1000
		else:
			return n_l + n_r + 10 * n_l * n_r - 4 * n_common + sort_diff


	def _difference(self, other):
		""" Compare the two ConfMetaData instances and rate how similar they are.
		"""
		return (self._set_diff(self.type_, other.type_),
				self._set_diff(self.organisers, other.organisers),
				self._list_diff(self.topic_keywords, other.topic_keywords),
				self._list_diff(self.qualifiers, other.qualifiers),
				self._set_diff(self.number, other.number)
		)


	def __str__(self):
		vals = []
		if self.topic_keywords:
			vals.append('topic=[' + ', '.join(self.topic_keywords) + ']')
		if self.organisers:
			vals.append('organisers={' + ', '.join(self.organisers) + '}')
		if self.number:
			vals.append('number={' + ', '.join(self.number) + '}')
		if self.type_:
			vals.append('type={' + ', '.join(self.type_) + '}')
		if self.qualifiers:
			vals.append('qualifiers={' + ', '.join(self.qualifiers) + '}')
		return ', '.join(vals)


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


	@classmethod
	def columns(cls):
		""" Return column titles for cfp data.
		"""
		return ['Acronym', 'Title', 'CORE 2017 Rank', 'Field']


	def values(self):
		""" What we'll show
		"""
		return [self.acronym, self.title, self.rank, self.field]


	def __eq__(self, other):
		return isinstance(other, self.__class__) and (self.rank, self.acronym, self.title, other.field) == (other.rank, other.acronym, other.title, other.field)


	def __lt__(self, other):
		return (self.ranksort(), self.acronym, self.title, other.field) < (other.ranksort(), other.acronym, other.title, other.field)


	def __str__(self):
		vals = ['{}={}'.format(s, getattr(self, s)) for s in self.__slots__ if getattr(self, s) not in {None, '(missing)'}]
		dat = super(Conference, self).__str__()
		if dat:
			vals.append(dat)
		return '{}({})'.format(type(self).__name__, ', '.join(vals))



class CallForPapers(ConfMetaData):
	_base_url = None
	_url_cfpsearch = None
	_url_cfpseries = None

	_date_fields = ['abstract', 'submission', 'notification', 'camera_ready', 'conf_start', 'conf_end']
	_date_names = ['Abstract Registration Due', 'Submission Deadline', 'Notification Due', 'Final Version Due', 'Start', 'End']

	__slots__ = ('conf', 'desc', 'dates', 'orig', 'url_cfp', 'year', 'link')


	def __init__(self, conf, year, desc = '', url_cfp = None, link = None, **kwargs):
		# Initialize parent parsing with the description
		super(CallForPapers, self).__init__(desc, conf.acronym, year, **kwargs)

		self.conf = conf
		self.desc = desc
		self.year = year
		self.dates = {}
		self.orig = {}
		self.link = link or '(missing)'
		self.url_cfp = url_cfp


	def extrapolate_missing_dates(self, prev_cfp):
		# NB: it isn't always year = this.year, e.g. the submission can be the year before the conference dates
		prev_dates = set(prev_cfp.dates.keys())
		dates = set(self.dates.keys())

		# direct extrapolations to year + 1
		for field in (field for field in {'conf_start', 'submission'} & prev_dates - dates):
			n = self._date_fields.index(field)
			self.dates[field] = prev_cfp.dates[field].replace(year = prev_cfp.dates[field].year + 1)

			self.orig[field] = False
			dates.add(field)

		# extrapolate by keeping
		extrapolate_from = {'conf_end': 'conf_start', 'camera_ready': 'conf_start', 'abstract': 'submission', 'notification': 'submission'}
		for field in (field for field in extrapolate_from.keys() & prev_dates - dates if extrapolate_from[field] in dates & prev_dates):
			self.dates[field] = self.dates[extrapolate_from[field]] + (prev_cfp.dates[field] - prev_cfp.dates[extrapolate_from[field]])

			self.orig[field] = False
			dates.add(field)


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
		dates_found = self.dates.keys()
		self.verify_conf_dates(dates_found)
		self.verify_submission_dates()


	def verify_conf_dates(self, dates_found):
		if {'conf_start', 'conf_end'} <= dates_found:
			err = []
			fix = 0
			s, e = (self.dates['conf_start'], self.dates['conf_end'])
			orig = '{} -- {}'.format(self.dates['conf_start'], self.dates['conf_end'])

			if s.year != self.year or e.year != self.year: # no conference over new year's eve, right?
				err.append('not in correct year')
				s, e = (s.replace(year = self.year), e.replace(year = self.year))
				fix += 1

			if e < s:
				err.append('end before start')
				try: # try flipping day and month
					flip = (s.replace(day = s.month, month = s.day), e.replace(day = e.month, month = e.day))
				except ValueError:
					flip = (0, 0)

				if flip[1] > flip[0] and flip[1] - flip[0] < datetime.timedelta(days = 10):
					s, e = flip
					fix += 1
				else:
					# if that's no good, just swap start and end
					s, e = (e, s)
					fix += 1

			if e - s > datetime.timedelta(days = 20):
				err.append('too far apart')
				try: # try flipping day and month
					flip = (s.replace(day = s.month, month = s.day), e.replace(day = e.month, month = e.day))
				except ValueError:
					flip = (0, 0)

				if flip[1] > flip[0] and flip[1] - flip[0] < datetime.timedelta(days = 10):
					s, e = flip
					fix += 1
				else:
					# cancel suggestion if at this stage it still is no good
					s, e = (self.dates['conf_start'], self.dates['conf_end'])

			if err:
				diag = 'Conferences dates {} are {} for {} {}'.format(orig, ' and '.join(err), self.conf.acronym, self.year)

				if len(err) == fix:
					Progress().clean_print('{}: using {} -- {} instead'.format(diag, s, e))
					# Use corrected dates, but take care to mark as guesses
					self.dates['conf_start'], self.dates['conf_end'] = (s, e)
					self.orig['conf_start'], self.orig['conf_end'] = (False, False)
				else:
					raise CFPCheckError(diag)


	def verify_submission_dates(self):
		pre_dates = {k:self.dates[k] for k in {'submission', 'abstract', 'notification'} & self.dates.keys()}
		if 'conf_start' in self.dates and pre_dates:
			err = []
			fix = 0

			for k, d in pre_dates.items():
				if d > self.dates['conf_start']:
					err.append('{} ({}) after conference'.format(k, d))
					if d.year == self.year:
						self.dates[k] = pre_dates[k].replace(year = self.year - 1)
						fix += 1
				elif self.dates['conf_start'] - d > datetime.timedelta(days = 365):
					err.append('{} ({}) too long before conference'.format(k, d))
					pass

			if err:
				diag = 'Conferences submission dates issues: {}, for {} {} ({} -- {})'.format(' and '.join(err), self.conf.acronym, self.year, self.dates['conf_start'], self.dates['conf_end'])

				if len(err) == fix:
					Progress().clean_print('{}: using {} instead'.format(diag, {k: str(self.dates[k]) for k in pre_dates}))
					for k in pre_dates:
						self.dates[k] = pre_dates[k]
						self.orig[k] = False
					# Use corrected dates, but take care to mark as guesses
				else:
					raise CFPCheckError(diag)


	@classmethod
	def find_link(cls, conf, year):
		""" Find the link to the conference page in the search page's soup
		"""
		search_f = 'cache/' + 'search_cfp_{}-{}.html'.format(conf.acronym, year).replace('/', '_')
		soup = get_soup(cls._url_cfpsearch, search_f, params = {'q': conf.acronym, 'y': year})

		options = (cls(conf, year, desc, url) for desc, url in cls.parse_search(conf, year, soup))

		try:
			return min((o for o in options if max(o.rating()) < 1000), key = lambda o: sum(o.rating()))
		except ValueError:
			raise CFPNotFoundError('No link with rating < 1000 for {} {}'.format(conf.acronym, year))


	@classmethod
	def get_cfp(cls, conf, year):
		""" Fetch the cfp from wiki-cfp for the given conference at the given year.
		"""
		try:
			cfp = cls.find_link(conf, year)
			cfp.fetch_cfp_data()
			return cfp

		except ConnectionError:
			raise CFPNotFoundError('Connection error when fetching CFP for {} {}'.format(conf.acronym, year))


	@classmethod
	def columns(cls):
		""" Return column titles for cfp data.
		"""
		return cls._date_names + ['orig_' + d for d in cls._date_fields] + ['Link', 'CFP url']


	def values(self):
		""" Return values of cfp data, in column order.
		"""
		return [self.dates.get(f, None) for f in self._date_fields] + [self.orig.get(f, None) for f in self._date_fields] + [self.link, self.url_cfp]


	def max_date(self):
		""" Get the max date in the cfp
		"""
		return max(self.dates.values())


	def rating(self):
		""" Rate the (in)adequacy of the cfp with its conference: lower is better.
		"""
		return self._difference(self.conf)[:4]


	def __str__(self):
		vals = ['{}={}'.format(s, getattr(self, s)) for s in self.__slots__ if s not in {'dates', 'orig'} and getattr(self, s) != None and getattr(self, s)  != '(missing)']
		if self.dates:
			vals.append('dates={' + ', '.join('{}:{}{}'.format(field, self.dates[field], '*' if not self.orig[field] else '') for field in self._date_fields if field in self.dates) + '}')
		dat = super(CallForPapers, self).__str__()
		if dat:
			vals.append(dat)
		return '{}({})'.format(type(self).__name__, ', '.join(vals))


class WikicfpCFP(CallForPapers):
	_base_url = 'http://www.wikicfp.com'
	_url_cfpsearch = urljoin(_base_url, '/cfp/servlet/tool.search')
	_url_cfpseries = urljoin(_base_url, '/cfp/series?t=c&i={initial}')
	_url_cfpevent  = urljoin(_base_url, '/cfp/servlet/event.showcfp') #?eventid={cfpid}
	_url_cfpevent_query = {'copyownerid': '90704'} # override some parameters


	@staticmethod
	def parse_date(d):
		return datetime.datetime.strptime(d, '%b %d, %Y').date()


	@classmethod
	def parse_confseries(cls, soup):
		""" Given the BeautifulSoup of a CFP series list page, generate all (acronym, description, url) tuples for links that
		point to conference series.
		"""
		links = soup.findAll('a', {'href': lambda l: l.startswith('/cfp/program')})
		return (tuple(l.parent.text.strip().split(' - ', 1)) + urljoin(cls._base_url, l['href']) for l in links)


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
			scheme, netloc, path, query, fragment = urlsplit(urljoin(cls._url_cfpevent, conf_link['href']))
			# update the query with cls._url_cfpevent_query
			query = urlencode({**parse_qs(query), **cls._url_cfpevent_query}, doseq = True)

			yield (conf_name[0], urlunsplit((scheme, netloc, path, query, fragment)))


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
	_core_file = 'core.csv'

	_historical = re.compile(r'\b(previous(ly)?|was|(from|pre) [0-9]{4}|merge[dr])\b', re.IGNORECASE)

	@classmethod
	@memoize
	def get_forcodes(cls):
		""" Fetch and return the mapping of For Of Research (FOR) codes to the corresponding names.
		"""
		forcodes = {}

		with open('for_codes.json', 'r') as f:
			forcodes = json.load(f)

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
	def _fetch_confs(cls):
		""" Generator of all conferences listed on the core site, as dicts
		"""
		# fetch page 0 outside loop to get page/result counts, will be in cache for loop access
		soup = get_soup(cls._url_corerank.format(0), 'cache/ranked_{}.html'.format(1))

		result_count_re = re.compile('Showing results 1 - ([0-9]+) of ([0-9]+)')
		result_count = soup.find(text = result_count_re)
		per_page, n_results = map(int, result_count_re.search(result_count).groups())
		pages = (n_results + per_page - 1) // per_page

		forcodes = cls.get_forcodes()

		with open(cls._core_file, 'w') as csv, Progress(operation = 'fetching CORE list', maxpos = n_results) as prog:
			print('title;acronym;rank;field', file=csv)

			for p in range(pages):
				f = 'cache/ranked_{}.html'.format(per_page * p + 1)
				soup = get_soup(cls._url_corerank.format(p), f)

				table = soup.find('table')
				rows = iter(table.findAll('tr'))

				headers = [r.text.strip().lower() for r in next(rows).findAll('th')]

				tpos = headers.index('title')
				apos = headers.index('acronym')
				rpos = headers.index('rank')
				fpos = headers.index('primary for')

				for row in prog.iterate(rows, p * per_page):
					val = [r.text.strip() for r in row.findAll('td')]
					yield Conference(cls.strip_trailing_paren(val[tpos]), val[apos], val[rpos], forcodes.get(val[fpos], None))


	@classmethod
	def _save_confs(cls, conflist):
		""" Save conferences to a file where we have the values cached cleanly.
		"""
		with open(cls._core_file, 'w') as csv:
			print('title;acronym;rank;field', file=csv)
			for conf in conflist:
				print(';'.join((conf.acronym, conf.title, conf.rank, conf.field)), file=csv)



	@classmethod
	def _load_confs(cls):
		""" Load conferences from a file where we have the values cached cleanly.
		"""
		f_age = datetime.datetime.fromtimestamp(os.stat(cls._core_file).st_mtime)
		if datetime.datetime.today() - f_age > datetime.timedelta(days = -1):
			raise FileNotFoundError('Cached file too old')

		with open(cls._core_file, 'r') as f:
			assert 'title;acronym;rank;field' == next(f).strip()
			confs = [l.strip().split(';') for l in f]

		with Progress(operation = 'loading CORE list', maxpos = len(confs)) as prog:
			return [Conference(*c) for c in prog.iterate(confs)]


	@classmethod
	def get_confs(cls):
		""" Generator of all conferences listed on the core site, as dicts
		"""
		try:
			return list(cls._load_confs())
		except FileNotFoundError:
			pass

		confs = list(uniq(cls._fetch_confs()))
		cls._save_confs(confs)

		return confs


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

	print('{"columns":', file=out);
	json.dump([{'title': c} for c in sum(([col + ' ' + str(y) for col in CallForPapers.columns()] for y in years), Conference.columns())], out)
	print(',\n"data": [', file=out)
	writing_first_conf = True

	core = CoreRanking.get_confs()
	with Progress(operation = 'fetching calls for papers') as prog:
		for conf in prog.iterate(core):
			values = conf.values()
			cfps_found = 0
			last_year = None
			for y in years:
				try:
					cfp = WikicfpCFP.get_cfp(conf, y)
					cfps_found += 1
					# possibly try other CFP providers?
				except CFPNotFoundError:
					cfp = None
				except CFPCheckError as e:
					cfp = None
					prog.clean_print(str(e))

				if not cfp:
					if last_year:
						cfp = CallForPapers(conf, y, desc = last_year.desc, link = last_year.link, url_cfp = last_year.url_cfp)
					else:
						cfp = CallForPapers(conf, y)

				if last_year:
					cfp.extrapolate_missing_dates(last_year)
				values += cfp.values()
				last_year = cfp

			if cfps_found:
				if not writing_first_conf: print(',', file=out)
				else: writing_first_conf = False

				# filter out empty values for non-date columns
				json.dump(values, out, default = json_encode_dates)

	scrape_date = datetime.datetime.fromtimestamp(min(os.path.getctime(f) for f in glob.glob('cache/cfp_*.html')))
	print(scrape_date.strftime('\n], "date":"%Y-%m-%d"}'), file=out)


if __name__ == '__main__':
	with open('cfp.json', 'w') as out:
		update_confs(out)


