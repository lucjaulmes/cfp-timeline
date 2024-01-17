#!/usr/bin/python3

import re
import os
import sys
import json
import glob
import time
import click
import shutil
import enchant
import inflection
import requests
import datetime
from functools import total_ordering
from collections import Counter
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qs, urlencode
from bs4 import BeautifulSoup


class CFPNotFoundError(Exception):
	pass


class CFPCheckError(Exception):
	pass


def clean_print(*args, **kwargs):
	""" Line print(), but first erase anything on the current line (e.g. a progress bar) """
	if args and kwargs.get('file', sys.stdout).isatty():
		if not hasattr(clean_print, '_clear_line'):
			clean_print._clear_line = f'\r{" " * shutil.get_terminal_size().columns}\r{{}}'
		args = (clean_print._clear_line.format(args[0]), *args[1:])
	print(*args, **kwargs)


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
		except RuntimeError:
			pass

		if n == 0:
			return self._ahead[0]
		else:
			return self._ahead[:n]


class RequestWrapper:
	""" Static wrapper of request.get() to implement caching and waiting between requests """
	last_req_times = {}
	use_cache = True
	delay = 0

	@classmethod
	def set_delay(cls, delay):
		cls.delay = delay

	@classmethod
	def set_use_cache(cls, use_cache):
		cls.use_cache = use_cache

	@classmethod
	def wait(cls, url):
		""" Wait until at least :attr:`~delay` seconds for the next same-domain request """
		key = urlsplit(url).netloc
		now = time.time()

		wait = cls.last_req_times.get(urlsplit(url).netloc, 0) + cls.delay - now
		cls.last_req_times[urlsplit(url).netloc] = now + max(0, wait)

		if wait >= 0:
			time.sleep(wait)


	@classmethod
	def get_soup(cls, url, filename, **kwargs):
		""" Simple caching mechanism. Fetch a page from url and save it in filename.

		If filename exists, return its contents instead.
		"""
		if cls.use_cache:
			try:
				with open(filename, 'r') as fh:
					return BeautifulSoup(fh.read(), 'lxml')
			except FileNotFoundError:
				pass

		cls.wait(url)
		r = requests.get(url, **kwargs)

		if cls.use_cache:
			with open(filename, 'w') as fh:
				print(r.text, file=fh)

		return BeautifulSoup(r.text, 'lxml')


def normalize(string):
	""" Singularize and lower casing of a word """
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

	_dict = enchant.Dict('EN_US')
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
		""" Compare the two ConfMetaData instances and rate how similar they are.  """
		return (self._set_diff(self.type_, other.type_),
				self._set_diff(self.organisers, other.organisers),
				self._list_diff(self.topic_keywords, other.topic_keywords),
				self._list_diff(self.qualifiers, other.qualifiers),
				self._set_diff(self.number, other.number)
		)


	def __str__(self):
		vals = []
		if self.topic_keywords:
			vals.append(f'topic=[{", ".join(self.topic_keywords)}]')
		if self.organisers:
			vals.append(f'organisers={{{", ".join(self.organisers)}}}')
		if self.number:
			vals.append(f'number={{{", ".join(self.number)}}}')
		if self.type_:
			vals.append(f'type={{{", ".join(self.type_)}}}')
		if self.qualifiers:
			vals.append(f'qualifiers={{{", ".join(self.qualifiers)}}}')
		return ', '.join(vals)


@total_ordering
class Conference(ConfMetaData):
	__slots__ = ('acronym', 'title', 'rank', 'ranksys', 'field')
	_ranks = {rk: num for num, rk in enumerate('A++ A* A+ A A- B B- C D E'.split())}  # unified for both sources, lower is better

	def __init__(self, title, acronym, rank=None, field=None, ranksys=None, **kwargs):
		super(Conference, self).__init__(title, acronym, **kwargs)

		self.title = title
		self.acronym = acronym
		self.ranksys = (ranksys,)
		self.rank = (rank or None,)
		self.field = field or '(missing)'


	def ranksort(self):
		""" Utility to sort the ranks based on the order we want (such ash A* < A).  """
		return min(self._ranks.get(rank, len(self._ranks)) for rank in self.rank)


	@classmethod
	def columns(cls):
		""" Return column titles for cfp data """
		return ['Acronym', 'Title', 'Rank system', 'Rank', 'Field']


	def values(self):
		""" What we'll show """
		return [self.acronym, self.title, self.ranksys, self.rank, self.field]


	def __eq__(self, other):
		return isinstance(other, self.__class__) and (self.acronym, self.title, self.rank, self.ranksys, self.field) == (other.acronym, other.title, other.rank, other.ranksys, other.field)


	def __lt__(self, other):
		return (self.acronym, self.title, self.ranksort(), self.ranksys, self.field) < (other.acronym, other.title, other.ranksort(), other.ranksys, other.field)


	def __str__(self):
		vals = ['{}={}'.format(slot, val) for slot, val in ((s, getattr(self, s)) for s in self.__slots__) if val != '(missing)']
		dat = super(Conference, self).__str__()
		if dat:
			vals.append(dat)
		return '{}({})'.format(type(self).__name__, ', '.join(vals))



class CallForPapers(ConfMetaData):
	_base_url = None
	_url_cfpsearch = None
	_url_cfpseries = None

	_date_fields = ['abstract', 'submission', 'notification', 'camera_ready', 'conf_start', 'conf_end']
	_date_names = ['Abstract Registration Due', 'Submission Deadline', 'Notification Due', 'Final Version Due', 'startDate', 'endDate']

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
			try:
				self.dates[field] = prev_cfp.dates[field].replace(year = prev_cfp.dates[field].year + 1)
			except ValueError:
				print(prev_cfp.dates[field], prev_cfp.dates[field].month, prev_cfp.dates[field].day)
				assert prev_cfp.dates[field].month == 2 and prev_cfp.dates[field].day == 29
				self.dates[field] = prev_cfp.dates[field].replace(year = prev_cfp.dates[field].year + 1, day = 28)

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


	def fetch_cfp_data(self):
		""" Parse a page from wiki-cfp. Return all useful data about the conference.  """
		f = 'cache/' + 'cfp_{}-{}_{}.html'.format(self.conf.acronym, self.year, self.conf.topic()).replace('/', '_') # topic('-')
		self.parse_cfp(RequestWrapper.get_soup(self.url_cfp, f))


	def verify_conf_dates(self):
		dates_found = self.dates.keys()

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
				diag = f'{self.conf.acronym} {self.year}: Conferences dates {orig} are {" and ".join(err)}'

				if len(err) == fix:
					# Use corrected dates, but take care to mark as guesses
					self.dates['conf_start'], self.dates['conf_end'] = (s, e)
					self.orig['conf_start'], self.orig['conf_end'] = (False, False)
					return f'{diag}: using {s} -- {e} instead'
				else:
					raise CFPCheckError(diag)


	def verify_submission_dates(self):
		pre_dates = {'submission', 'abstract', 'notification', 'camera_ready'} & set(self.dates.keys())
		typical_delays = {key: (datetime.timedelta(lo), datetime.timedelta(hi)) for key, (lo, hi) in {
			'abstract': (95, 250),
			'camera_ready': (0, 120),
			'notification': (20, 150),
			'submission': (40, 250),
		}.items()}

		if 'conf_start' in self.dates and pre_dates:
			err = []
			uncorrected = set()
			corrected = set()

			for k, d in [(k, self.dates[k]) for k in pre_dates]:
				delay = self.dates['conf_start'] - d
				if delay < datetime.timedelta(0):
					err.append('{} ({}) after conference'.format(k, d))
				elif delay > datetime.timedelta(days=365):
					err.append('{} ({}) too long before conference'.format(k, d))
				else:
					continue

				# If shifting the year gets us into the “typical” delay, use that date and mark as a guess
				# Typically for conf at year Y, all dates are set at year Y even if they should be previous year.
				shifted = d.replace(year=d.year + int(delay.days // 365.2425))
				lo, hi = typical_delays.get(k)
				if hi >= self.dates['conf_start'] - shifted >= lo:
					self.dates[k] = shifted
					self.orig[k] = False
					corrected.add(k)
				else:
					err[-1] += f' (shifted: {(self.dates["conf_start"] - shifted).days}d)'
					# delete uncorrectable camera ready dates to avoid raising an error
					if k == 'camera_ready':
						self.dates.pop('camera_ready')
						corrected.add(k)
					else:
						uncorrected.add(k)

			if err:
				diag = f'{self.conf.acronym} {self.year} ({self.dates["conf_start"]} -- {self.dates["conf_end"]}): '\
					   f'Submission dates issues: {" and ".join(err)}'

				if not uncorrected:
					return f'{diag}: using {", ".join(f"{k}={self.dates.get(k)}" for k in corrected)} instead'
				else:
					raise CFPCheckError(diag)


	@classmethod
	def find_link(cls, conf, year, debug=False):
		""" Find the link to the conference page in the search page

		Have parse_search extract links from the page's soup, then compute a rating for each and keep the best (lowest).
		Use the amount of missing ("TBD") fields as a tie breaker.
		"""
		search_f = 'cache/' + 'search_cfp_{}-{}.html'.format(conf.acronym, year).replace('/', '_')
		soup = RequestWrapper.get_soup(cls._url_cfpsearch, search_f, params = {'q': conf.acronym, 'year': year})

		# Rating of 1000 disqualifies.
		best_candidate = None
		best_score = (1000, 1000)

		for desc, url, missing in cls.parse_search(conf, year, soup):
			candidate = cls(conf, year, desc, url)
			rating = candidate.rating()
			if debug:
				print(f'[{rating}] {candidate}')
			if max(rating) < 1000 and best_score > (sum(rating), missing):
				best_candidate = candidate
				best_score = (sum(rating), missing)

		if not best_candidate:
			raise CFPNotFoundError('No link with rating < 1000 for {} {}'.format(conf.acronym, year))
		else:
			return best_candidate


	@classmethod
	def get_cfp(cls, conf, year, debug=False):
		""" Fetch the cfp from wiki-cfp for the given conference at the given year.  """
		try:
			cfp = cls.find_link(conf, year, debug=debug)
			cfp.fetch_cfp_data()
			return cfp

		except requests.exceptions.ConnectionError:
			raise CFPNotFoundError('Connection error when fetching CFP for {} {}'.format(conf.acronym, year))


	@classmethod
	def columns(cls):
		""" Return column titles for cfp data.  """
		return cls._date_names + ['orig_' + d for d in cls._date_fields] + ['Link', 'CFP url']


	def values(self):
		""" Return values of cfp data, in column order.  """
		return [self.dates.get(f, None) for f in self._date_fields] + [self.orig.get(f, None) for f in self._date_fields] + [self.link, self.url_cfp]


	def max_date(self):
		""" Get the max date in the cfp """
		return max(self.dates.values())


	def rating(self):
		""" Rate the (in)adequacy of the cfp with its conference: lower is better.  """
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
		# some ISO 8601 or RFC 3339 format
		return datetime.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S').date()


	@classmethod
	def parse_confseries(cls, soup):
		""" Given the BeautifulSoup of a CFP series list page, generate all (acronym, description, url) tuples for links that
		point to conference series.
		"""
		links = soup.find_all('a', {'href': lambda l: l.startswith('/cfp/program')})
		return (tuple(l.parent.text.strip().split(' - ', 1)) + urljoin(cls._base_url, l['href']) for l in links)


	@classmethod
	def parse_search(cls, conf, year, soup):
		""" Given the BeautifulSoup of a CFP search page, generate all (description, url) tuples for links that seem
		to correspond to the conference and year requested.
		"""
		search = '{} {}'.format(conf.acronym, year).lower()
		for conf_link in soup.find_all('a', href=True, text=lambda t: t and ' '.join(t.lower().strip().split()) == search):
			# find links name "acronym year" and got to first parent <tr>
			for tr in conf_link.parents:
				if tr.name == 'tr':
					break
			else:
				raise ValueError('Cound not find parent row!')

			# first row has 2 td tags, one contains the link, the other the description. Get the one not parent of the link.
			conf_name = [td.text for td in tr.find_all('td') if td not in conf_link.parents]
			scheme, netloc, path, query, fragment = urlsplit(urljoin(cls._url_cfpevent, conf_link['href']))
			# update the query with cls._url_cfpevent_query. Sort the parameters to minimize changes across versions.
			query = urlencode(sorted({**parse_qs(query), **cls._url_cfpevent_query}.items()), doseq = True)

			# next row has the dates and location, count how many of those are not defined yet
			while tr:
				tr = tr.next_sibling
				if tr.name == 'tr':
					break
			else:
				raise ValueError('Cound not find dates row!')

			missing_info = [td.text for td in tr.find_all('td')].count('TBD')

			yield (conf_name[0], urlunsplit((scheme, netloc, path, query, fragment)), missing_info)


	@classmethod
	def _find_xmlns_attrs(cls, attr, tag):
		return attr.startswith('xmlns:') and ('rdf.data-vocabulary.org' in tag[attr] or 'purl.org/dc/' in tag[attr])


	def parse_cfp(self, soup):
		""" Given the BeautifulSoup of the CFP page, update self.dates and self.link

		WikiCFP has all info nicely porcelain-ish formatted in some RDF and Dublin Core xmlns tags.
		Extract {key: val} data from one of:
		<tag property="${xmlns_prefix}${key}" content="${val}"></tag>
		<tag property="${xmlns_prefix}${key}">${val}</tag>
		"""

		metadata = {}
		for xt in soup.find_all(lambda tag: any(self._find_xmlns_attrs(attr, tag) for attr in tag.attrs)):
			xmlns_attr = next(attr for attr in xt.attrs if self._find_xmlns_attrs(attr, xt))
			xmlns_pfx = xmlns_attr[len('xmlns:'):] + ':'

			xt_data = {xt['property'][len(xmlns_pfx):]: xt['content'] if xt.has_attr('content') else xt.text for xt \
						in xt.find_all(property = lambda val: type(val) is str and val.startswith(xmlns_pfx))}

			if 'purl.org/dc/' in xt[xmlns_attr]:
				metadata.update(xt_data)

			elif xt_data.keys() == {'summary', 'startDate'}:
				# this is a pair of tags that contain just a date, use summary value as key
				metadata[xt_data['summary']] = self.parse_date(xt_data['startDate'])

			elif xt_data.get('eventType', None) == 'Conference':
				# Remove any clashes with DC's values, which are cleaner
				metadata.update({key:self.parse_date(val) if key.endswith('Date') else val \
								for key, val in xt_data.items() if key not in metadata})

			else:
				print('Error: unexpected RDF or DC data: {}'.format(xt_data))

		for f, name in zip(self._date_fields, self._date_names):
			try:
				self.dates[f] = metadata[name]
				self.orig[f] = True
			except KeyError:
				pass # Missing date in data

		# source is the URL, it's sometimes empty
		if 'source' in metadata and metadata['source']:
			self.link = metadata['source']


class Ranking(object):
	_historical = re.compile(r'\b(previous(ly)?|was|(from|pre) [0-9]{4}|merge[dr])\b', re.IGNORECASE)

	@classmethod
	def get_confs(cls):
		""" Generator of all conferences listed in a source, as dicts """
		try:
			return list(cls._load_confs())
		except FileNotFoundError:
			return cls.update_confs()


	@classmethod
	def strip_trailing_paren(cls, string):
		""" If string ends with a parenthesized part, remove it, e.g. "foo (bar)" -> "foo" """
		string = string.strip()
		try:
			paren = string.index(' (')
			if string[-1] == ')' and cls._historical.search(string[paren + 2:-1]):
				return string[:paren]
		except ValueError:
			pass
		return string


	@classmethod
	def merge(cls, confs_a, confs_b):
		dict_a = {}
		dict_b = {}
		for conf in confs_a:
			dict_a.setdefault(conf.acronym.upper(), []).append(conf)
		for conf in confs_b:
			dict_b.setdefault(conf.acronym.upper(), []).append(conf)

		common = set(dict_a.keys()) & set(dict_b.keys())
		merged = [conf for k in set(dict_a) - common for conf in dict_a[k]] \
			   + [conf for k in set(dict_b) - common for conf in dict_b[k]]
		for acronym in common:
			list_a = dict_a.pop(acronym)
			list_b = dict_b.pop(acronym)
			cmp = [[1000 for _ in list_b] for _ in list_a]
			for n, conf_a in enumerate(list_a):
				for m, conf_b in enumerate(list_b):
					cmp[n][m] = sum(conf_a._difference(conf_b))
			while list_a and list_b and min(map(min, cmp)) < 1000:
				rowmins = [min(row) for row in cmp]
				match_a = rowmins.index(min(rowmins))
				match_b = cmp[match_a].index(min(rowmins))

				merge_pair = [list_a[match_a], list_b[match_b]]
				conf = merge_pair[0]
				conf.rank = (*list_a[match_a].rank, *list_b[match_b].rank)
				conf.ranksys = (*list_a[match_a].ranksys, *list_b[match_b].ranksys)
				merged.append(conf)

				cmp = [row[:match_b] + row[match_b + 1:] for row in cmp]
				del cmp[match_a], list_a[match_a], list_b[match_b]

			merged.extend(list_a)
			merged.extend(list_b)

		print(f'Merged conferences {len(confs_a)} + {len(confs_b)} = {len(merged)} total + {len(confs_a) + len(confs_b) - len(merged)} in common')
		return merged


class GGSRanking(Ranking):
	_url_ggsrank = 'https://scie.lcc.uma.es/gii-grin-scie-rating/conferenceRating.jsf'
	_ggs_file = 'ggs.csv'

	@classmethod
	def _load_confs(cls):
		""" Load conferences from a file where we have the values cached cleanly.  """
		f_age = datetime.datetime.fromtimestamp(os.stat(cls._ggs_file).st_mtime)
		if datetime.datetime.today() - f_age > datetime.timedelta(days=365):
			raise FileNotFoundError('Cached file too old')

		with open(cls._ggs_file, 'r') as f:
			assert 'acronym;title;rank' == next(f).strip()
			confs = [l.strip().split(';') for l in f]

		with click.progressbar(confs, label='loading GGS list…') as prog:
			return [Conference(cls.strip_trailing_paren(tit), acr, rat, None, 'GGS2021') for acr, tit, rat in prog]

	@classmethod
	def update_confs(cls):
		soup = RequestWrapper.get_soup(cls._url_ggsrank, 'cache/gii-grin-scie-rating_conferenceRating.html')
		link = soup.find('a', attrs={'href': lambda dest: dest.split(';jsessionid=')[0].endswith('.xlsx')}).attrs['href']
		file_url = urljoin(cls._url_ggsrank, link)

		import pandas as pd, csv
		df = pd.read_excel(file_url, header=1, usecols=['Title', 'Acronym', 'GGS Rating'])\
			   .rename(columns={'GGS Rating': 'rank', 'Title': 'title', 'Acronym': 'acronym'})

		# Drop old stuff or no acronyms (as they are used for lookup)
		df = df[~(df['rank'].str.contains('discontinued|now published as journal', case=False) | df['acronym'].isna() | df['acronym'].str.len().eq(0))]

		ok_rank = df['rank'].str.match('^[A-Z][+-]*$')
		print('Non-standard ratings:')
		print(df['rank'].mask(ok_rank).value_counts().to_string())
		df['rank'] = df['rank'].where(ok_rank)
		df['title'] = df['title'].str.replace(';', ',').str.title().str.replace(r'\b(Acm|Ieee)\b', lambda m: m[1].upper(), regex=True)\
				.str.replace(r'\b(On|And|In|Of|For|The|To|Its)\b', lambda m: m[1].lower(), regex=True)

		sort_ranks = lambda ser: ser.map(Conference._ranks).fillna(len(Conference._ranks)) if ser.name == 'rank' else ser
		col_order = ['acronym', 'title', 'rank']
		df[col_order].sort_values(by=col_order, key=sort_ranks).to_csv(cls._ggs_file, sep=';', index=False, quoting=csv.QUOTE_NONE)


class CoreRanking(Ranking):
	""" Utility class to scrape CORE conference listings and generate `~Conference` objects.  """
	_url_corerank = 'http://portal.core.edu.au/conf-ranks/?search=&by=all&source=CORE{}&sort=arank&page={}'
	_year = 2023
	_core_file = 'core.csv'
	_for_file = 'for_codes.json'

	@classmethod
	def _fetch_confs(cls):
		""" Internal generator of all conferences listed on the core site, as dicts """
		# fetch page 1 outside loop to get page/result counts, will be in cache for loop access
		soup = RequestWrapper.get_soup(cls._url_corerank.format(cls._year, 1), 'cache/ranked_{1}.html')
		ranking = f'CORE{cls._year}'

		result_count_re = re.compile('Showing results 1 - ([0-9]+) of ([0-9]+)')
		result_count = soup.find(string=result_count_re)
		per_page, n_results = map(int, result_count_re.search(result_count).groups())
		pages = (n_results + per_page - 1) // per_page

		with open(cls._for_file, 'r') as f:
			forcodes = json.load(f)
		non_standard_ranks = Counter()

		with click.progressbar(label='fetching CORE list…', length=n_results) as prog:
			for p in range(1, pages + 1):
				soup = RequestWrapper.get_soup(cls._url_corerank.format(cls._year, p), f'cache/ranked_{p}.html')

				table = soup.find('table')
				rows = iter(table.find_all('tr'))

				headers = [' '.join(r.text.split()).lower() for r in next(rows).find_all('th')]

				tpos = headers.index('title')
				apos = headers.index('acronym')
				rpos = headers.index('rank')
				fpos = headers.index('primary for')

				for row in rows:
					val = [' '.join(r.text.split()) for r in row.find_all('td')]
					acronym, title, rank, code = val[apos], val[tpos], val[rpos], val[fpos]
					# Some manual corrections applied to the CORE database:
					# - ISC changed their acronym to "ISC HPC"
					# - Searching cfps for Euro-Par finds EuroPar, but not the other way around
					if acronym == 'ISC' and cls.strip_trailing_paren(title) == 'ISC High Performance':
						acronym += ' HPC'
					elif acronym == 'EuroPar' and (cls.strip_trailing_paren(title) ==
									  'International European Conference on Parallel and Distributed Computing'):
						acronym = 'Euro-Par'

					# Also normalize rankings
					if rank.startswith('National') or rank.startswith('Regional'):
						place = rank[8:].strip("(): -").title()
						rank = (f'{rank[:8]}{": " if place else ""}'
									 f'{"USA" if place == "Usa" else "Korea" if place == "S. korea" else place}')
					elif not re.match(r'^(Australasian )?[A-Z]\*?$', rank):
						non_standard_ranks[rank] += 1
						rank = None

					yield Conference(cls.strip_trailing_paren(title), acronym, rank, forcodes.get(code, None), ranking)
					prog.update(1)

		# Manually add some missing conferences from previous year data.
		manual = [
			('MICRO',     'International Symposium on Microarchitecture',                                           'A',  '4601', 'CORE2018'),
			('VLSI',      'Symposia on VLSI Technology and Circuits',                                               'A',  '4009', 'CORE2018'),
			('ICC',       'IEEE International Conference on Communications',                                        'B',  '4006', 'CORE2018'),
			('IEEE RFID', 'IEEE International Conference on Radio Frequency Identification',                        'B',  '4006', 'CORE2018'),
			('M2VIP',     'Mechatronics and Machine Vision in Practice',                                            'B',  '4611', 'CORE2018'),
			('ICASSP',    'IEEE International Conference on Acoustics, Speech and Signal Processing',               'B',  '4006', 'CORE2018'),
			('RSS',       'Robotics: Science and Systems',                                                          'A*', '4611', 'CORE2018'),
			('BuildSys',  'ACM International Conference on Systems for Energy-Efficient Built Environments',        'A',  '4606', 'CORE2018'),
			('DAC',       'Design Automation Conference',                                                           'A',  '4606', 'CORE2018'),
			('FSR',       'International Conference on Field and Service Robotics',                                 'A',  '4602', 'CORE2018'),
			('CDC',       'IEEE Conference on Decision and Control',                                                'A',  '4009', 'CORE2018'),
			('ASAP',      'International Conference on Application-specific Systems, Architectures and Processors', 'A',  '4606', 'CORE2018'),
			('ISR',       'International Symposium on Robotics',                                                    'A',  '4007', 'CORE2018'),
			('ISSCC',     'IEEE International Solid-State Circuits Conference',                                     'A',  '4009', 'CORE2018'),

			('RANDOM',	   'International Workshop on Randomization and Computation',								'A',  '4613', 'CORE2021'),
			('SIMULTECH',  'International Conference on Simulation and Modeling Methodologies, Technologies '
						   'and Applications',																		'C',  '4606', 'CORE2021'),
			('ICCS',       'International Conference on Conceptual Structures',										'B',  '4613', 'CORE2021'),
		]
		for acronym, name, rank, code, ranking in manual:
			yield Conference(name, acronym, rank, forcodes.get(code, None), ranking)

		if non_standard_ranks:
			print('Non-standard ratings:')
			width = max(map(len, non_standard_ranks.keys())) + 3
			for key, num in non_standard_ranks.most_common():
				print(f'{key:{width}} {num}')


	@classmethod
	def _save_confs(cls, conflist):
		""" Save conferences to a file where to cache them.  """
		with open(cls._core_file, 'w') as csv:
			print('acronym;title;ranksys;rank;field', file=csv)
			for conf in conflist:
				try:
					print(';'.join((conf.acronym, conf.title, conf.ranksys[0], conf.rank[0] or '', conf.field)), file=csv)
				except TypeError:
					print(conf.acronym, conf.title, conf.ranksys, conf.rank, conf.field, file=sys.stderr)
					raise



	@classmethod
	def _load_confs(cls):
		""" Load conferences from a file where we have the values cached cleanly.  """
		f_age = datetime.datetime.fromtimestamp(os.stat(cls._core_file).st_mtime)
		if datetime.datetime.today() - f_age > datetime.timedelta(days = 1):
			raise FileNotFoundError('Cached file too old')

		with open(cls._core_file, 'r') as f:
			assert 'acronym;title;ranksys;rank;field' == next(f).strip()
			confs = [l.strip().split(';') for l in f]

		with click.progressbar(confs, label='loading CORE list…') as prog:
			return [Conference(cls.strip_trailing_paren(tit), acr, rnk, fld, sys) for acr, tit, sys, rnk, fld in prog]


	@classmethod
	def update_confs(cls):
		""" Refresh and make a generator of all conferences listed on the core site, as dicts """
		conf_list = []
		for conf in sorted(cls._fetch_confs()):
			if not conf_list or conf != conf_list[-1]:
				conf_list.append(conf)
		cls._save_confs(conf_list)

		return conf_list


	@classmethod
	def get_confs(cls):
		""" Generator of all conferences listed on the core site, as dicts """
		try:
			return list(cls._load_confs())
		except FileNotFoundError:
			return cls.update_confs()


def json_encode_dates(obj):
	if isinstance(obj, datetime.date):
		return str(obj)
	else:
		raise TypeError('{} not encodable'.format(obj))


@click.group(invoke_without_command=True, chain=True)
@click.option('--cache/--no-cache', default=True, help='Cache files in ./cache')
@click.option('--delay', type=float, default=0, help='Delay between requests to the same domain')
@click.pass_context
def update(ctx, cache, delay):
	""" Update the Core-CFP data. If no command is provided, update_confs is run.  """
	RequestWrapper.set_delay(delay)
	RequestWrapper.set_use_cache(cache)

	if not ctx.invoked_subcommand:
		# Default is to_update calls for papers
		cfps()


@update.command()
def core():
	""" Update the cached list of CORE conferences """
	CoreRanking.update_confs()


@update.command()
def ggs():
	""" Update the cached list of GII-GRIN-SCIE (GGS) conferences """
	GGSRanking.update_confs()


@update.command()
@click.option('--out', default='cfp.json', help='Output file for CFPs', type=click.File('w'))
@click.option('--debug/--no-debug', default=False, help='Show debug output')
def cfps(out, debug=False):
	""" Update the calls for papers from the conference lists  """
	today = datetime.datetime.now().date()
	# use years from 6 months ago until next year
	years = range((today - datetime.timedelta(days = 366 / 2)).year, (today + datetime.timedelta(days = 365)).year + 1)

	print('{{"years": {}, "columns":'.format([y for y in years if y >= today.year]), file=out);
	json.dump(sum(([col + ' ' + str(y) for col in CallForPapers.columns()] for y in years if y >= today.year), Conference.columns()), out)
	print(',\n"data": [', file=out)
	writing_first_conf = True

	confs = sorted(Ranking.merge(CoreRanking.get_confs(), GGSRanking.get_confs()))

	progressbar = click.progressbar(confs, label='fetching calls for papers…', update_min_steps=len(confs) // 1000 if not RequestWrapper.delay else 1,
									item_show_func=lambda conf: f'{conf.acronym} {conf.title}' if conf is not None else '')

	with open('parsing_errors.txt', 'w') as errlog, progressbar as conf_iterator:
		for conf in conf_iterator:
			values = conf.values()
			cfps_found = 0
			last_year = None
			for y in years:
				if debug:
					clean_print(f'Looking up CFP {conf} {y}')
				try:
					cfp = WikicfpCFP.get_cfp(conf, y, debug=debug)

					err = cfp.verify_conf_dates()
					if err:
						clean_print(err)
						print(err.replace(':', ';', 1) + ';' + cfp.url_cfp + ';corrected', file=errlog)

					err = cfp.verify_submission_dates()
					if err:
						clean_print(err)
						print(err.replace(':', ';', 1) + ';' + cfp.url_cfp + ';corrected', file=errlog)

					cfps_found += 1
					# possibly try other CFP providers?

				except CFPNotFoundError as e:
					if debug:
						print(f'> {e}\n')
					cfp = None
				except CFPCheckError as e:
					if debug:
						print(f'> {e}\n')
					else:
						clean_print(e)
					print(str(e).replace(':', ';', 1) + ': no satisfying correction heuristic;' + cfp.url_cfp + ';ignored', file=errlog)
					cfp = None
				else:
					if debug:
						print('> Found\n')

				if not cfp:
					if last_year:
						cfp = CallForPapers(conf, y, desc = last_year.desc, link = last_year.link, url_cfp = last_year.url_cfp)
					else:
						cfp = CallForPapers(conf, y)

				if last_year:
					cfp.extrapolate_missing_dates(last_year)
				if y >= today.year:
					values += cfp.values()
				last_year = cfp

			if cfps_found:
				if not writing_first_conf: print(',', file=out)
				else: writing_first_conf = False

				# filter out empty values for non-date columns
				json.dump(values, out, default = json_encode_dates)

	try:
		scrape_date = datetime.datetime.fromtimestamp(min(os.path.getctime(f) for f in glob.glob('cache/cfp_*.html')))
	except ValueError:
		scrape_date = datetime.datetime.now()
	print(scrape_date.strftime('\n], "date":"%Y-%m-%d"}'), file=out)


if __name__ == '__main__':
	update()
