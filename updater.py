#!/usr/bin/python3

from __future__ import annotations

import re
import os
import sys
import csv
import copy as copy_
import json
import glob
import time
import click
import shutil
import enchant
import inflection
import requests
import datetime
import operator
import functools
import numpy as np
import pandas as pd
from urllib import parse
import bs4
import warnings

from typing import cast, ClassVar, Generic, Match, Mapping, overload, TextIO, TypeVar
from collections.abc import ItemsView, Iterable, Iterator, Generator, MutableMapping

_term_columns = shutil.get_terminal_size().columns

# Annoying irrelevant pandas warning from .str.contains
warnings.filterwarnings('ignore', 'This pattern is interpreted as a regular expression, and has match groups')
# We’re parsing some xhtml
warnings.filterwarnings('ignore', "It looks like you're parsing an XML document using an HTML parser.")

class CFPNotFoundError(Exception):
	pass


class CFPCheckError(Exception):
	pass


def clean_print(*args, **kwargs):
	""" Line print(), but first erase anything on the current line (e.g. a progress bar) """
	if args and kwargs.get('file', sys.stdout).isatty():
		if not hasattr(clean_print, '_clear_line'):
			clean_print._clear_line = f'\r{" " * _term_columns}\r{{}}'
		args = (clean_print._clear_line.format(args[0]), *args[1:])
	print(*args, **kwargs)


T = TypeVar('T')

class PeekIter(Generic[T]):
	""" Iterator that allows

	Attributes:
		_it: wrapped by this iterator
		_ahead: stack of the next elements to be returned by __next__
	"""
	_it: Iterator[T]
	_ahead: list[T]

	def __init__(self, iterable: Iterator[T]):
		super(PeekIter, self)
		self._it = iterable
		self._ahead = []


	def __iter__(self) -> Iterator[T]:
		return self


	def __next__(self) -> T:
		if self._ahead:
			return self._ahead.pop(0)
		else:
			return next(self._it)


	@overload
	def peek(self) -> T:
		...

	@overload
	def peek(self, n: int) -> list[T]:
		...

	def peek(self, n: int = 0) -> T | list[T]:
		""" Returns next element(s) that will be returned from the iterator.

		Args:
			n: Number of positions to look ahead in the iterator.

		Returns:
			The next element if n = 0, or the list of the up to n next elements if n > 0.

		Raises:
			IndexError: There are no further elements (only if n = 0).
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
	last_req_times: dict[str, float] = {}
	use_cache: bool = True
	delay: float = 0

	@classmethod
	def set_delay(cls, delay: float):
		cls.delay = delay

	@classmethod
	def set_use_cache(cls, use_cache: bool):
		cls.use_cache = use_cache

	@classmethod
	def wait(cls, url: str):
		""" Wait until at least :attr:`~delay` seconds for the next same-domain request """
		key = parse.urlsplit(url).netloc
		now = time.time()

		wait = cls.last_req_times.get(parse.urlsplit(url).netloc, 0) + cls.delay - now
		cls.last_req_times[parse.urlsplit(url).netloc] = now + max(0, wait)

		if wait >= 0:
			time.sleep(wait)


	@classmethod
	def get_soup(cls, url: str, filename: str, **kwargs) -> bs4.BeautifulSoup:
		""" Simple caching mechanism. Fetch a page from url and save it in filename.

		If filename exists, return its contents instead.
		kwargs are forwarded to :func:`requests.get`
		"""
		if cls.use_cache:
			try:
				with open(filename, 'r') as fh:
					return bs4.BeautifulSoup(fh.read(), 'lxml')
			except FileNotFoundError:
				pass

		cls.wait(url)
		r = requests.get(url, **kwargs)

		if cls.use_cache:
			with open(filename, 'w') as fh:
				print(r.text, file=fh)

		return bs4.BeautifulSoup(r.text, 'lxml')


def normalize(string: str) -> str:
	""" Singularize and lower casing of a word """
	# Asia -> Asium and Meta -> Metum, really?
	return inflection.singularize(string.lower()) if len(string) > 3 else string.lower()


class ConfMetaData:
	""" Heuristic to reduce a conference title to a matchable set of words. """
	# separators in an acronum
	_sep = re.compile(r'[-_/ @&,.]+')

	# associations, societies, institutes, etc. that organize conferences
	_org = {
		'ACIS': 'Association for Computer and Information Science', 'ACL': 'Association for Computational Linguistics',
		'ACM': 'Association for Computing Machinery', 'ACS': 'Arab Computer Society', 'AoM': 'Academy of Management',
		'CSI': 'Computer Society of Iran', 'DIMACS': 'Center for Discrete Mathematics and Theoretical Computer Science',
		'ERCIM': 'European Research Consortium for Informatics and Mathematics', 'Eurographics': 'Eurographics',
		'Euromicro': 'Euromicro', 'IADIS': 'International Association for the Development of the Information Society',
		'IAPR': 'International Association for Pattern Recognition',
		'IAVoSS': 'International Association for Voting Systems Sciences', 'ICSC': 'ICSC Interdisciplinary Research',
		'IEEE': 'Institute of Electrical and Electronics Engineers',
		'IET': 'Institution of Engineering and Technology',
		'IFAC': 'International Federation of Automatic Control',
		'IFIP': 'International Federation for Information Processing',
		'IMA': 'Institute of Mathematics and its Applications', 'KES': 'KES International',
		'MSRI': 'Mathematical Sciences Research Institute', 'RSJ': 'Robotics Society of Japan',
		'SCS': 'Society for Modeling and Simulation International',
		'SIAM': 'Society for Industrial and Applied Mathematics',
		'SLKOIS': 'State Key Laboratory of Information Security',
		'SIGOPT': 'DMV Special Interest Group in Optimization',
		'SIGNLL': 'ACL Special Interest Group in Natural Language Learning',
		'SPIE': 'International Society for Optics and Photonics',
		'TC13': 'IFIP Technical Committee on Human–Computer Interaction',
		'Usenix': 'Advanced Computing Systems Association', 'WIC': 'Web Intelligence Consortium',
	}

	# ACM Special Interest Groups
	_sig = {
		'ACCESS': 'Accessible Computing', 'ACT': 'Algorithms Computation Theory', 'Ada': 'Ada Programming Language',
		'AI': 'Artificial Intelligence', 'APP': 'Applied Computing', 'ARCH': 'Computer Architecture',
		'BED': 'Embedded Systems', 'Bio': 'Bioinformatics', 'CAS': 'Computers Society',
		'CHI': 'Computer-Human Interaction', 'COMM': 'Data Communication', 'CSE': 'Computer Science Education',
		'DA': 'Design Automation', 'DOC': 'Design Communication', 'ecom': 'Electronic Commerce',
		'EVO': 'Genetic Evolutionary Computation', 'GRAPH': 'Computer Graphics Interactive Techniques',
		'HPC': 'High Performance Computing', 'IR': 'Information Retrieval', 'ITE': 'Information Technology Education',
		'KDD': 'Knowledge Discovery Data', 'LOG': 'Logic Computation', 'METRICS': 'Measurement Evaluation',
		'MICRO': 'Microarchitecture', 'MIS': 'Management Information Systems',
		'MOBILE': 'Mobility Systems, Users, Data Computing', 'MM': 'Multimedia', 'MOD': 'Management Data',
		'OPS': 'Operating Systems', 'PLAN': 'Programming Languages', 'SAC': 'Security, Audit Control',
		'SAM': 'Symbolic Algebraic Manipulation', 'SIM': 'Simulation Modeling', 'SOFT': 'Software Engineering',
		'SPATIAL': 'SIGSPATIAL', 'UCCS': 'University College Computing Services', 'WEB': 'Hypertext Web',
		'ART': 'Artificial Intelligence', # NB ART was renamed AI
	}

	_meeting_types = {'congress', 'conference', 'consortium', 'seminar', 'symposium', 'workshop', 'tutorial'}
	_qualifiers = {
		'american', 'asian', 'australasian', 'australian', 'annual', 'biennial', 'european', 'iberoamerican',
		'international', 'joint', 'national',
	}
	_replace = {
		 # shortenings
		'intl': 'international', 'conf': 'conference', 'dev': 'development',
		 # americanize
		'visualisation': 'visualization', 'modelling': 'modeling', 'internationalisation': 'internationalization',
		'defence': 'defense', 'standardisation': 'standardization', 'organisation': 'organization',
		'optimisation': 'optimization,', 'realising': 'realizing', 'centre': 'center',
		# encountered typos
		'syste': 'system', 'computi': 'computing', 'artifical': 'artificial', 'librari': 'library',
		'databa': 'database', 'conferen': 'conference', 'bioinformatic': 'bioinformatics', 'symposi': 'symposium',
		'evoluti': 'evolution', 'proce': 'processes', 'provi': 'proving', 'techology': 'technology',
		'bienniel': 'biennial', 'entertainme': 'entertainment', 'retriev': 'retrieval', 'engineeri': 'engineering',
		'sigraph': 'siggraph', 'intelleligence': 'intelligence', 'simululation': 'simulation',
		'inteligence': 'intelligence', 'manageme': 'management', 'applicatio': 'application',
		'developme': 'development', 'cyberworl': 'cyberworld', 'scien': 'science', 'personalizati': 'personalization',
		'computati': 'computation', 'implementati': 'implementation', 'languag': 'language', 'traini': 'training',
		'servic': 'services', 'intenational': 'international', 'complexi': 'complexity', 'storytelli': 'storytelling',
		'measureme': 'measurement', 'comprehensi': 'comprehension', 'synthe': 'synthesis', 'evaluatin': 'evaluation',
		'intermational': 'international', 'internaltional': 'international', 'interational': 'international',
		'technologi': 'technology', 'applciation': 'application',
	}

	# NB simple acronym management, only works while first word -> acronym mapping is unique
	_acronyms = {''.join(word[0] for word in acr.split()): [normalize(word) for word in acr.split()] for acr in [
		'call for papers', 'geographic information system', 'high performance computing', 'message passing interface',
		'object oriented', 'operating system', 'parallel virtual machine', 'public key infrastructure',
		'special interest group',
	]}
	# Computer Performance Evaluation ? Online Analytical Processing: OLAP? aspect-oriented programming ?

	_tens = {'twenty', 'thirty', 'fourty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'}
	_ordinal = re.compile(r'[0-9]+(st|nd|rd|th)|(({tens})?(first|second|third|(four|fif|six|seven|eigh|nine?)th))|'
					      r'(ten|eleven|twelf|(thir|fourt|fif|six|seven|eigh|nine?)teen)th'
						  .format(tens = '|'.join(_tens)))

	_sigcmp = {normalize(f'SIG{group}'): group for group in _sig}
	_orgcmp = {normalize(entity): entity for entity in _org}

	_acronym_start = {words[0]: acr for acr, words in _acronyms.items()}
	_sig_start = {normalize(desc.split()[0]): group for group, desc in _sig.items() if group != 'ART'}

	_dict = enchant.DictWithPWL('EN_US', 'dict.txt')
	_misspelled: dict[str, list[tuple[str, ...]]] = {}

	acronym_words: list[str]
	topic_keywords: list[str]
	organisers: set[str]
	number: set[str]
	type_: set[str]
	qualifiers: list[str]

	def __init__(self, title: str, conf_acronym: str, year: str | int = ''):
		super().__init__()

		self.acronym_words = self._sep.split(conf_acronym.lower())
		self.topic_keywords = []
		self.organisers = set()
		self.number = set()
		self.type_ = set()
		self.qualifiers = []

		self.classify_words(title, normalize(conf_acronym), str(year))


	def classify_words(self, string: str, *ignored: str):
		# lower case, replace characters in dict by whitepace, repeated spaces will be removed by split()
		normalized = (normalize(w) for w in string.translate({ord(c): ' ' for c in "-/&,():_~'\".[]|*@"}).split())
		words = PeekIter(w for w in normalized
						 if w not in {'', 'the', 'on', 'for', 'of', 'in', 'and', 'its', *ignored, ''.join(ignored)})

		# semantically filter conference editors/organisations, special interest groups (sig...), etc.
		for w in words:
			# Only manually fix typos, afraid of over-correcting
			try:
				w = self._replace[w]
			except KeyError:
				pass

			if w in self._orgcmp:
				self.organisers.add(self._orgcmp[w])
				continue

			if w in self._meeting_types:
				self.type_.add(w)
				continue

			# TODO: match call types, possibly using peekiter to distinguish between multi-word expressions
			# call for posters, call for [full] papers, phd symposium, etc.

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
					for _ in next_words:
						next(words)
					continue

				# TODO some acronyms have special characters, e.g. A/V, which means they appear as 2 words
			except (KeyError, IndexError):
				pass

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

				except (KeyError, IndexError):
					pass

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
				except IndexError:
					pass

			m = ConfMetaData._ordinal.match(w)
			if m:
				self.number.add(m.group(0))
				continue

			# anything surviving to this point surely describes the topic of the conference
			self.topic_keywords.append(w)

			# Log words marked as incorrect in misspellings - ad-hoc ignored words can be used as conf identifiers
			if self._dict and not self._dict.check(w):
				if self._dict.check(w.capitalize()) or self._dict.check(w.title()):
					pass
				elif w.endswith('um') and self._dict.check(f'{w[:-2]}a'.capitalize()):
					pass  # inflection overcorrects (often proper) nouns asia -> asium, malaysia -> malaysium, etc.
				elif w not in (normalize(s).replace('-', '') for s in self._dict.suggest(w)):
					self._misspelled.setdefault(w, []).append((*ignored, string))


	def topic(self, sep: str = ' ') -> str:
		return sep.join(self.topic_keywords).title()


	@classmethod
	def _set_diff(cls, left: set[str], right: set[str], require_common: bool = True) -> float:
		""" Return an int quantifying the difference between the sets. Lower is better.

		Penalize a bit for difference on a single side, more for differences on both sides, under the assumption that
		partial information on one side is better than dissymetric information
		"""
		n_common = len(set(left) & set(right))
		l = len(left) - n_common
		r = len(right) - n_common

		if require_common and l and r and not n_common:
			return np.inf
		else:
			return  l + r + 10 * l * r - 2 * n_common


	@classmethod
	def _list_diff(cls, left: list[str], right: list[str], require_common: bool = True) -> float:
		""" Return a float quantifying the difference between the lists of words.

		Uset the same as `~set_diff` and add penalties for dfferences in word order.
		"""
		# for 4 diffs => 4 + 0 -> 5, 3 + 1 -> 8, 2 + 2 -> 9
		common = set(left) & set(right)
		n_common = len(common)
		n_l, n_r = len(left) - n_common, len(right) - n_common
		l = [w for w in left if w in common]
		r = [w for w in right if w in common]
		try:
			mid = round(sum(l.index(c) - r.index(c) for c in common) / len(common))
			sort_diff = sum(abs(l.index(c) - r.index(c) - mid) for c in common) / n_common
		except ZeroDivisionError:
			sort_diff = 0

		# disqualify if there is nothing in common
		if require_common and left and right and not common:
			return np.inf
		else:
			return n_l + n_r + 10 * n_l * n_r - 4 * n_common + sort_diff


	@classmethod
	def _acronym_diff(cls, left: list[str], right: list[str]) -> float:
		""" Return a float quantifying the difference between lists of words in acronyms.

		More specific than _list_diff as we always want the first word to be an exact match,
		but we may reinterpret contiguous words as a single one.
		"""
		if left == right:
			return -40 * len(left) * len(right)

		# Special case se we don’t match e.g. IFIP SEC with IFIP-DSS: ignore leading word if it’s an organizer
        # Note that an organiser name can be a conference name, e.g. usenix
		left_org = len(left) > 1 and left[0] in cls._orgcmp
		right_org = len(right) > 1 and right[0] in cls._orgcmp
		# If both sides start with the same org name, compare ignoring it. Discount 2 (half a common word).
		if left_org and right_org and left[0] == right[0]:
			return cls._acronym_diff(left[1:], right[1:]) - 2
		# If only one side starts with an org name, compare ignoring it unless it matches the name on the other side
		# Add 1 (penalty for a dissymetric word)
		elif left_org or right_org and left[0] != right[0]:
			return cls._acronym_diff(left[int(left_org):], right[int(right_org):]) + 1

		left_prefixes = {''.join(left[:n + 1]): n for n in range(len(left))}
		right_prefixes = {''.join(right[:n + 1]): n for n in range(len(right))}

		common = left_prefixes.keys() & right_prefixes.keys()
		if not common:
			return np.inf

		prefix = max(common, key=len)
		nsep_left_prefix = left_prefixes[prefix]
		nsep_right_prefix = right_prefixes[prefix]

		# penalty of 1 per ignored separator, 10 per ignored word
		return nsep_left_prefix + nsep_right_prefix + 10 * cls._list_diff([prefix, *left[nsep_left_prefix + 1:]],
																		  [prefix, *right[nsep_right_prefix + 1:]])


	def _difference(self, other: ConfMetaData) -> tuple[float, float, float, float, float, float]:
		""" Compare the two ConfMetaData instances and rate how similar they are.  """
		return (
			self._acronym_diff(self.acronym_words, other.acronym_words),
			self._set_diff(self.type_, other.type_),
			self._set_diff(self.organisers, other.organisers),
			self._list_diff(self.topic_keywords, other.topic_keywords),
			self._list_diff(self.qualifiers, other.qualifiers, require_common=False) / 2,
			self._set_diff(self.number, other.number)
		)


	def str_info(self) -> list[str]:
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
		return vals


	def __repr__(self) -> str:
		return f'{type(self).__name__}({", ".join(self.str_info())})'


@functools.total_ordering
class Conference(ConfMetaData):
	__slots__ = ('acronym', 'title', 'rank', 'ranksys', 'field')
	# unified ranks from all used sources, lower is better
	_ranks: ClassVar[dict[str, int]] = {rk: num for num, rk in enumerate('A++ A* A+ A A- B B- C D E'.split())}

	title: str
	acronym: str
	ranksys: tuple[str | None, ...]
	rank: tuple[str | None, ...]
	field: str

	def __init__(self, acronym: str, title: str, rank: str | None = None, ranksys: str | None = None,
				 field: str | None = None, **kwargs: int | str):
		super(Conference, self).__init__(title, acronym, **kwargs)

		self.title = title
		self.acronym = acronym
		self.ranksys = (ranksys if pd.notna(ranksys) else None,)
		self.rank = (rank if pd.notna(rank) else None,)
		self.field = field or '(missing)'


	def ranksort(self) -> int:
		""" Utility to sort the ranks based on the order we want (such ash A* < A).  """
		return min(len(self._ranks) if rank is None else self._ranks.get(rank, len(self._ranks)) for rank in self.rank)


	@classmethod
	def columns(cls) -> list[str]:
		""" Return column titles for cfp data """
		return ['Acronym', 'Title', 'Rank', 'Rank system', 'Field']


	def values(self, sort: bool = False) -> tuple[str, str, int | tuple[str | None, ...], tuple[str | None, ...], str]:
		""" What we'll show """
		return (self.acronym, self.title, self.ranksort() if sort else self.rank, self.ranksys, self.field)


	@classmethod
	def from_series(cls, series: pd.Series[str]) -> Conference:
		""" Convert from a series """
		return cls(series['acronym'], series['title'], series['rank'], series['ranksys'], series['field'])


	@classmethod
	def merge(cls, left: Conference, right: Conference) -> Conference:
		new = copy_.copy(left)
		# In case we have matched different acronyms, keep the version with most separations/words
		if len(left.acronym_words) < len(right.acronym_words):
			new.acronym, new.acronym_words = right.acronym, right.acronym_words
		new.rank = left.rank + right.rank
		new.ranksys = left.ranksys + right.ranksys
		return new


	def __eq__(self, other: object) -> bool:
		if not isinstance(other, self.__class__):
			return NotImplemented
		return self.values() == other.values()


	def __lt__(self, other: Conference) -> bool:
		return self.values(True) < other.values(True)


	def str_info(self) -> list[str]:
		vals = [f'{slot}={getattr(self, slot)}' for slot in self.__slots__ if getattr(self, slot) != '(missing)']
		vals.extend(super().str_info())
		return vals


class Dates(MutableMapping[str, T]):
	__slots__ = ('abstract', 'submission', 'notification', 'camera_ready', 'conf_start', 'conf_end')

	def __getitem__(self, key: str) -> T:
		try:
			return getattr(self, key)
		except AttributeError:
			raise KeyError(key) from None

	def __setitem__(self, key: str, value: T):
		setattr(self, key, value)

	def __delitem__(self, key: str):
		delattr(self, key)

	def __len__(self) -> int:
		return sum(hasattr(self, attr) for attr in self.__slots__)

	def __iter__(self) -> Iterator[str]:
		return iter(attr for attr in self.__slots__ if hasattr(self, attr))

	def items(self) -> ItemsView[str, T]:
		for attr in self.__slots__:
			try:
				val = getattr(self, attr)
			except AttributeError:
				pass
			else:
				yield attr, val


class CallForPapers(ConfMetaData):
	_date_names = (
		'Abstract Registration Due', 'Submission Deadline', 'Notification Due', 'Final Version Due', 'startDate',
		'endDate',
	)
	_typical_delays = {
		'abstract': (95, 250),
		'camera_ready': (0, 120),
		'notification': (20, 150),
		'submission': (40, 250),
	}

	__slots__ = ('acronym', 'desc', 'dates', 'orig', 'url_cfp', 'year', 'link', 'id', 'date_errors')

	_url_cfpsearch: ClassVar[str]
	_fill_id: ClassVar[int] = sys.maxsize
	_cache: ClassVar[dict[int, CallForPapers]] = {}
	_errors: ClassVar[list] = []

	empty_series: ClassVar[pd.Series] = pd.Series(None, index=__slots__)

	acronym: str
	id: int
	desc: str
	year: int
	dates: Dates[datetime.datetime]
	orig: Dates[bool]
	link: str
	url_cfp: str | None
	date_errors: bool | None

	def __init__(self, acronym: str, year: int | str, id_: int, desc: str = '',
				 url_cfp: str | None = None, link: str | None = None):
		# Initialize parent parsing with the description
		super().__init__(desc, acronym, year)

		self.acronym = acronym
		self.id = id_
		self.desc = desc
		self.year = int(year)
		self.dates = Dates()
		self.orig = Dates()
		self.link = link or '(missing)'
		self.url_cfp = url_cfp
		self.date_errors = None


	@classmethod
	def build(cls, acronym: str, year: int | str, id_: int | None = None, desc: str = '',
			   url_cfp: str | None = None, link: str | None = None):
		cfp_id = CallForPapers._fill_id if id_ is None else id_
		try:
			return CallForPapers._cache[cfp_id]
		except KeyError:
			pass

		cfp = cls(acronym, year, cfp_id, desc, url_cfp, link)
		CallForPapers._cache.update({cfp_id: cfp})

		if id_ is None:
			CallForPapers._fill_id -= 1

		return cfp


	@classmethod
	def all_built_cfps(cls) -> Mapping[int, CallForPapers]:
		return cls._cache


	def extrapolate_missing(self, prev_cfp: CallForPapers | None):
		if pd.isna(prev_cfp) or prev_cfp is None:
			return self

		# NB: it isn't always year = this.year, e.g. the submission can be the year before the conference dates
		year_shift = self.year - prev_cfp.year
		assert year_shift > 0, 'Should only extrapolate from past conferences'

		if self.link == '(missing)':
			self.link = prev_cfp.link

		if self.url_cfp is None:
			self.url_cfp = prev_cfp.url_cfp

		# direct extrapolations to previous cfp + year_shift
		for field in ('conf_start', 'submission'):
			if field in self.dates or field not in prev_cfp.dates:
				continue
			n = Dates.__slots__.index(field)
			try:
				self.dates[field] = prev_cfp.dates[field].replace(year=prev_cfp.dates[field].year + year_shift)
			except ValueError:
				assert prev_cfp.dates[field].month == 2 and prev_cfp.dates[field].day == 29
				self.dates[field] = prev_cfp.dates[field].replace(year=prev_cfp.dates[field].year + year_shift, day=28)

			self.orig[field] = False

		# extrapolate by keeping offset with other date
		extrapolate_from = {
			'conf_start': {'conf_end', 'camera_ready'},
			'submission': {'abstract', 'notification'},
		}
		for orig, fields in extrapolate_from.items():
			if orig not in self.dates or orig not in prev_cfp.dates:
				continue
			for field in (fields - self.dates.keys()) & prev_cfp.dates.keys():
				self.dates[field] = self.dates[orig] + (prev_cfp.dates[field] - prev_cfp.dates[orig])
				self.orig[field] = False

		return self


	@classmethod
	def _parse_search(cls, conf: Conference, year: int | str,
				     soup: bs4.BeautifulSoup) -> Iterator[tuple[str, str, int, str, int]]:
		""" Generate the list of conferences from a search page.

		Yields:
			Info on the cfp from the search page: (acronym, name, unique id, url, number of missing dates/fields)
		"""
		raise NotImplementedError


	def _parse_cfp(self, soup: bs4.BeautifulSoup):
		""" Load the cfp infos from the page soup """
		raise NotImplementedError


	def fetch_cfp_data(self, debug: bool = False):
		""" Parse a page from online source. Load all useful data about the conference. """
		if self.date_errors is not None:
			return self
		self.date_errors = False

		assert self.url_cfp is not None, 'By definition of a check and a fetched cfp'

		f = f'cache/cfp_{self.acronym.replace("/", "_")}-{self.year}-{self.id}.html'
		self._parse_cfp(RequestWrapper.get_soup(self.url_cfp, f))

		try:
			if warn := self.verify_conf_dates():
				clean_print(warn)
				CallForPapers._errors.append(f'{warn.replace(":", ";", 1)};{self.url_cfp};corrected')

		except CFPCheckError as err:
			clean_print(f'> {err}' if debug else err)
			CallForPapers._errors.append(
				f'{str(err).replace(":", ";", 1)}: no satisfying correction heuristic;{self.url_cfp};ignored'
			)
			self.date_errors = True

		try:
			if warn := self.verify_submission_dates():
				clean_print(warn)
				CallForPapers._errors.append(f'{warn.replace(":", ";", 1)};{self.url_cfp};corrected')

		except CFPCheckError as err:
			clean_print(f'> {err}' if debug else err)
			CallForPapers._errors.append(
				f'{str(err).replace(":", ";", 1)}: no satisfying correction heuristic;{self.url_cfp};ignored'
			)
			self.date_errors = True

		return self


	@classmethod
	def _flip_day_month(cls, start: datetime.date, end: datetime.date):
		""" Fix a classic error of writing mm-dd-yyyy instead of dd-mm-yyyy by flipping day and month

		raises:
			ValueError: Invalid dates (typically a day was over 12), or resulting dates don’t make sense
		"""
		flip_start = start.replace(day=start.month, month=start.day)
		flip_end = end.replace(day=end.month, month=end.day)

		if flip_start > flip_end or flip_end >= flip_start + datetime.timedelta(days=10):
			raise ValueError('Resulting dates not fitting for conference interval')

		return flip_start, flip_end


	def verify_conf_dates(self) -> str | None:
		""" Check coherence of conference start and end dates

		returns:
			A message describing fixed issues, if any

		raises:
			CPFCheckError: An error with no satisfying correction heuristic was encountered
		"""
		if not {'conf_start', 'conf_end'} <= self.dates.keys():
			return None

		err = []
		nfixes = 0
		start, end = self.dates['conf_start'], self.dates['conf_end']

		# Assuming no conference over new year's eve, so year should match with both dates
		if start.year != self.year or end.year != self.year:
			err.append('not in correct year')
			start, end = start.replace(year=self.year), end.replace(year=self.year)
			nfixes += 1

		if end < start:
			err.append('end before start')
			try:
				start, end = self._flip_day_month(start, end)
			except ValueError:
				# if that'start no good, just swap start and end
				end, start = start, end
			nfixes += 1

		if end - start > datetime.timedelta(days=20):
			err.append('too far apart')
			try:
				start, end = self._flip_day_month(start, end)
			except ValueError:
				pass  # Don’t increment nfixes
			else:
				nfixes += 1

		if not err:
			return None

		diag = (f'{self.acronym} {self.year} ({self.dates["conf_start"]} -- {self.dates["conf_end"]}): '
				f'Conferences dates are {" and ".join(err)}')

		if nfixes < len(err):
			raise CFPCheckError(diag)

		# Use corrected dates, taking care to mark as guesses
		self.dates.update({'conf_start': start, 'conf_end': end})
		self.orig['conf_start'] = self.orig['conf_end'] = False
		return f'{diag}: using {start} -- {end} instead'


	def verify_submission_dates(self, delete_on_err: set[str] = {'camera_ready'}) -> str | None:
		""" Check coherence of submission dates, in terms of delay from a deadline to conference start

		returns:
			A message describing fixed issues, if any

		raises:
			CPFCheckError: An error with no satisfying correction heuristic was encountered
		"""
		if 'conf_start' not in self.dates or not self._typical_delays.keys() & self.dates.keys():
			return None

		err = []
		uncorrected = set()
		corrected = dict()

		start = self.dates['conf_start']

		for name, deadline in ((name, date) for name, date in self.dates.items() if name in self._typical_delays):
			# Only check that deadlines happen in the year before the conference start
			delay = (start - deadline).days
			if delay < 0:
				err.append(f'{name} ({deadline}) after conference start')
			elif delay > 365:
				err.append(f'{name} ({deadline}) too long before conference')
			else:
				continue

			# If shifting the year gets us into the “typical” delay, use that date and mark as a guess
			# Typical mistake for a conf in the first half of year Y, all dates are reported as year Y
			# even if they should be previous year.
			shifted = deadline.replace(year=deadline.year + int(delay // 365.2425))
			shifted_delay = start - shifted

			lo, hi = self._typical_delays[name]
			if hi >= shifted_delay.days >= lo:
				corrected[name] = shifted
			else:
				err.append(f'{err.pop()} (shifted: {shifted_delay.days}d)')
				uncorrected.add(name)

		if not err:
			return None

		diag = (f'{self.acronym} {self.year} ({self.dates["conf_start"]} -- {self.dates["conf_end"]}): '
			    f'Submission dates issues: {" and ".join(err)}')

		if uncorrected - delete_on_err:
			raise CFPCheckError(diag)

		# update with shifted dates and delete uncorrectable camera ready dates to avoid raising an error
		self.dates.update(corrected)
		self.orig.update({name: False for name in corrected})
		delete_keys = uncorrected & delete_on_err
		for key in delete_keys:
			del self.dates[key]

		fixes = [*(f'{name}={date}' for name, date in corrected.items()), *(f'no {name}' for name in delete_keys)]
		return f'{diag}: using {", ".join(fixes)} instead'


	@classmethod
	def find_link(cls, conf: Conference, year: int | str, debug: bool = False) -> tuple[CallForPapers, float, int]:
		""" Find the link to the conference page in the search page

		Have parse_search extract links from the page's soup, then compute a rating for each and keep the best (lowest).
		Use the amount of missing ("TBD") fields as a tie breaker.

		raises:
			CFPNotFoundError: No satisfying link was found on the search page
		"""
		search_f = f'cache/search_cfp_{conf.acronym.replace("/", "_")}-{year}.html'
		soup = RequestWrapper.get_soup(cls._url_cfpsearch, search_f, params = {'q': conf.acronym, 'year': year})

		cfp_list = []

		for acronym, desc, id_, url, missing in cls._parse_search(conf, year, soup):
			candidate = cls.build(acronym, year, id_, desc, url)
			rating = candidate.rating(conf)
			if debug:
				print(f'[{rating}] {candidate}')
			total_rating = sum(rating)
			if np.isfinite(total_rating):
				cfp_list.append([total_rating, *rating, missing, candidate])

		if not cfp_list:
			raise CFPNotFoundError(f'No link with acceptable rating for {conf.acronym} {year}')

		cfps = pd.DataFrame(cfp_list, columns=['rating', 'acronym', 'type', 'org', 'topic', 'qualif', 'missing', 'cfp'])
		return tuple(cfps.loc[cfps['rating'].idxmin(), ['cfp', 'rating', 'missing']])


	@classmethod
	def get_cfp(cls, conf: Conference, year: int | str, debug: bool = False) -> tuple[CallForPapers, float, int]:
		""" Fetch the cfp from wiki-cfp for the given conference at the given year.  """
		try:
			cfp, cmp, miss = cls.find_link(conf, year, debug=debug)
			cfp.fetch_cfp_data(debug=debug)
			return cfp, cmp, miss

		except requests.exceptions.ConnectionError:
			raise CFPNotFoundError('Connection error when fetching CFP for {} {}'.format(conf.acronym, year))


	@classmethod
	def columns(cls) -> list[str]:
		""" Return column titles for cfp data.  """
		return list(cls._date_names) + ['orig_' + d for d in Dates.__slots__] + ['Link', 'CFP url']


	def values(self) -> list[datetime.datetime | bool | str | None]:
		""" Return values of cfp data, in column order.  """
		return [*(self.dates.get(f, None) for f in Dates.__slots__),
				*(self.orig.get(f, None) for f in Dates.__slots__), self.link, self.url_cfp]


	def max_date(self) -> datetime.datetime:
		""" Get the max date in the cfp """
		return max(self.dates.values())


	def rating(self, conf: Conference) -> tuple[float, float, float, float, float]:
		""" Rate the (in)adequacy of the cfp with the given conference: lower is better. """
		# Just drop number (e.g. 34th intl conf...) comparison.
		return self._difference(conf)[:-1]


	def str_info(self) -> list[str]:
		vals = ['{}={}'.format(attr, getattr(self, attr)) for attr in self.__slots__
			    if attr not in {'dates', 'orig'} and (getattr(self, attr, None) or '(missing)') != '(missing)']
		if self.dates:
			vals.append('dates={' + ', '.join(f"{field}:{self.dates[field]}{'*' if not self.orig[field] else ''}"
											  for field in Dates.__slots__ if field in self.dates) + '}')
		return vals


class WikicfpCFP(CallForPapers):
	_base_url = 'http://www.wikicfp.com'
	_url_cfpsearch = parse.urljoin(_base_url, '/cfp/servlet/tool.search')
	_url_cfpevent  = parse.urljoin(_base_url, '/cfp/servlet/event.showcfp') #?eventid={cfpid}
	_url_cfpevent_query = {'copyownerid': ['90704']} # override some parameters


	@classmethod
	def _parse_date(cls, dt: str) -> datetime.date:
		# some ISO 8601 or RFC 3339 format
		return datetime.datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S').date()


	@classmethod
	def _parse_search(cls, conf: Conference, year: int | str,
					 soup: bs4.BeautifulSoup) -> Iterator[tuple[str, str, int, str, int]]:
		""" Given the BeautifulSoup of a CFP search page, generate all infos for links that seem to correspond to the
		conference and year requested.

		Yields:
			Info on the cfp from the search page: (acronym, name, unique id, url, number of missing dates/fields)
		"""
		test_words = ConfMetaData._sep.split(conf.acronym.lower())
		def match_acronym(text):
			if text is None:
				return False
			try:
				*words, cfp_year = ConfMetaData._sep.split(text.lower().strip())
				cfp_year = int(cfp_year)
			except ValueError:
				return False
			if year != cfp_year or not words:
				return False
			return 100 > ConfMetaData._acronym_diff(test_words, words)

		for conf_link in soup.find_all('a', href=True, string=match_acronym):
			# find links name "acronym year" and got to first parent <tr>
			for tag in conf_link.parents:
				if tag.name == 'tr':
					tr = tag
					break
			else:
				raise ValueError('Cound not find parent row!')

			acronym = ' '.join(conf_link.text.strip().split()[:-1])

			# first row has 2 td tags, one contains the link, the other the description. Get the non-parent of the link.
			for td in tr.find_all('td'):
				if td not in conf_link.parents:
					conf_name = td.text.strip()
					break
			else:
				raise ValueError('Could not find conference name')

			scheme, netloc, path, query, fragment = parse.urlsplit(parse.urljoin(cls._url_cfpevent, conf_link['href']))
			query_dict = parse.parse_qs(query)
			try:
				id_ = int(query_dict['eventid'][0])
			except ValueError:
				raise ValueError('Could not find valid identifier in url') from None

			# update the query with cls._url_cfpevent_query. Sort the parameters to minimize changes across versions.
			query = parse.urlencode(sorted({**query_dict, **cls._url_cfpevent_query}.items()), doseq=True)

			# next row has the dates and location, count how many of those are not defined yet
			while tr:
				tr = tr.next_sibling
				if tr.name == 'tr':
					break
			else:
				raise ValueError('Cound not find dates row!')

			missing_info = [td.text for td in tr.find_all('td')].count('TBD')

			yield (acronym, conf_name, id_, parse.urlunsplit((scheme, netloc, path, query, fragment)), missing_info)


	@classmethod
	def _find_xmlns_attrs(cls, attr: str, tag: bs4.Tag) -> bool:
		return attr.startswith('xmlns:') and ('rdf.data-vocabulary.org' in tag[attr] or 'purl.org/dc/' in tag[attr])


	def _parse_cfp(self, soup: bs4.BeautifulSoup):
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

			xt_data = {xt['property'][len(xmlns_pfx):]: xt['content'] if xt.has_attr('content') else xt.text
					   for xt in xt.find_all(property=lambda val: type(val) is str and val.startswith(xmlns_pfx))}

			if 'purl.org/dc/' in xt[xmlns_attr]:
				metadata.update(xt_data)

			elif xt_data.keys() == {'summary', 'startDate'}:
				# this is a pair of tags that contain just a date, use summary value as key
				metadata[xt_data['summary']] = self._parse_date(xt_data['startDate'])

			elif xt_data.get('eventType', None) == 'Conference':
				# Remove any clashes with DC's values, which are cleaner
				metadata.update({key: self._parse_date(val) if key.endswith('Date') else val
								 for key, val in xt_data.items() if key not in metadata})

			else:
				print('Error: unexpected RDF or DC data: {}'.format(xt_data))

		for f, name in zip(Dates.__slots__, self._date_names):
			try:
				self.dates[f] = metadata[name]
				self.orig[f] = True
			except KeyError:
				pass  # Missing date in data

		# source is the URL, it's sometimes empty
		if 'source' in metadata and metadata['source']:
			self.link = metadata['source'].strip()


class Ranking:
	_historical = re.compile(r'\b(previous(ly)?|was|(from|pre|in) [0-9]{4}|merge[dr])\b', re.IGNORECASE)
	_col_order: ClassVar[list[str]]
	_file: ClassVar[str]

	@classmethod
	def get_confs(cls) -> pd.Series[Conference]:
		""" Generator of all conferences listed in a source, as a series of Conference objects """
		try:
			confs = cls._load_confs()
		except FileNotFoundError:
			confs = cls.update_confs()
		return confs.agg(Conference.from_series, axis='columns')


	@classmethod
	def update_confs(cls) -> pd.DataFrame:
		""" Refresh and make a generator of all conferences listed on the core site, as dicts """
		conf_list = cls._fetch_confs()
		cls._save_confs(conf_list)
		return conf_list


	@classmethod
	def _fetch_confs(cls) -> pd.DataFrame:
		""" Fetch unparsed conference info from online """
		raise NotImplementedError


	@classmethod
	def _save_confs(cls, confs: pd.DataFrame):
		""" Save unparsed conference info as a local cache/csv file """
		confs[cls._col_order].to_csv(cls._file, sep=';', index=False, quoting=csv.QUOTE_NONE)


	@classmethod
	def _load_confs(cls) -> pd.DataFrame:
		""" Load unparsed conference info from a local cache/csv file """
		f_age = datetime.datetime.fromtimestamp(os.stat(cls._file).st_mtime)
		if datetime.datetime.today() - f_age > datetime.timedelta(days=365):
			raise FileNotFoundError('Cached file too old')

		confs = pd.read_csv(cls._file, sep=';')
		assert confs.columns.symmetric_difference(cls._col_order).empty
		return confs


	@classmethod
	def strip_trailing_paren(cls, series: pd.Series[str]) -> pd.Series[str]:
		""" Strip parenthesized part at end of string, if it contains historical info (“was x”, ”pre 2019” etc.) """
		series = series.str.strip()
		split_paren = series.str.split(' (', n=1, expand=True, regex=False)
		replace = series.str[-1].eq(')') & split_paren[1].str.contains(cls._historical, na=False)
		return series.mask(replace, split_paren[0])


	@classmethod
	def merge(cls, *confs: pd.Series[Conference], debug: list[str] | bool = False) -> pd.Series[Conference]:
		merged_confs, *confs_to_merge = confs
		for conf in confs_to_merge:
			merged_confs = cls._merge(merged_confs, conf, debug=debug)

		print(f'Merged conferences {" + ".join(str(len(series)) for series in confs)} = {len(merged_confs)} total'
			  f' + {sum(map(len, confs)) - len(merged_confs)} in common')
		return merged_confs


	@classmethod
	def _merge(cls, confs_a: pd.Series[Conference], confs_b: pd.Series[Conference],
			   debug: list[str] | bool = False) -> pd.Series[Conference]:
		""" Merge 2 sources of conferences into a single one, merging duplicate conferences and keeping unique ones. """
		# Mapping match-acronym to conference-id
		idx_a = pd.Series(confs_a.index, index=confs_a.map(operator.attrgetter('acronym')).str.upper(), name='id')
		idx_b = pd.Series(confs_b.index, index=confs_b.map(operator.attrgetter('acronym')).str.upper(), name='id')

		# Add for several words (possibly dash or slash-separated) the first-word variant and joined variants
		# So for Euro-Par we’ll check EuroPar and Euro
		multi_word_a = idx_a.index[idx_a.index.str.contains(ConfMetaData._sep)].to_series().str.split(ConfMetaData._sep)
		idx_a = pd.concat([
			idx_a,
			idx_a.loc[multi_word_a.index].rename(index=multi_word_a.str.join('').to_dict()),
			idx_a.loc[multi_word_a.index].rename(index=multi_word_a.str[0].to_dict()),
		])
		multi_word_b = idx_b.index[idx_b.index.str.contains(ConfMetaData._sep)].to_series().str.split(ConfMetaData._sep)
		idx_b = pd.concat([
			idx_b,
			idx_b.loc[multi_word_b.index].rename(index=multi_word_b.str.join('')),
			idx_b.loc[multi_word_b.index].rename(index=multi_word_b.str[0]),
		])

		common = idx_a.index.drop_duplicates().intersection(idx_b.index.drop_duplicates())
		if not len(common):
			return pd.concat([confs_a, confs_b], ignore_index=True)

		# Build all the pairs of elements we want to compare
		compared_pairs = pd.concat(ignore_index=True, objs=[
			pd.merge(idx_a[[acronym]], idx_b[[acronym]], how='cross', suffixes=('_a', '_b')) for acronym in common
		])
		# Compute the scores
		compared_pairs['score'] = compared_pairs.agg(
			lambda row: sum(ConfMetaData._difference(confs_a[row['id_a']], confs_b[row['id_b']])),
			axis='columns'
		)

		merged_id_dfs = []
		unmerged_a, unmerged_b, all_compared_pairs = confs_a, confs_b, compared_pairs
		while compared_pairs.size:
			# Drop scores marked as invalid
			best_matches = compared_pairs[~compared_pairs['score'].transform(np.isinf)]
			# Best (a, b) pairs for each a
			best_matches = best_matches.loc[best_matches.groupby('id_a')['score'].idxmin()]
			# Refine by best (a, b) pairs for each b; we don’t want to merge one b with several a
			best_matches = best_matches.loc[best_matches.groupby('id_b')['score'].idxmin()]

			if not best_matches.size:
				break

			# Drop the merged ids from conf lists and future comparisons
			compared_pairs = compared_pairs[~compared_pairs['id_a'].isin(best_matches['id_a']) &
											~compared_pairs['id_b'].isin(best_matches['id_b'])]
			unmerged_a = unmerged_a.drop(index=best_matches['id_a'])
			unmerged_b = unmerged_b.drop(index=best_matches['id_b'])

			merged_id_dfs.append(best_matches.drop(columns=['score']))

		if not len(merged_id_dfs):
			return pd.concat([confs_a, confs_b], ignore_index=True)

		merged_ids = pd.concat(merged_id_dfs)
		merged = pd.concat(ignore_index=True, objs=[unmerged_a, unmerged_b, merged_ids.agg(
			lambda row: Conference.merge(confs_a[row['id_a']], confs_b[row['id_b']]),
			axis='columns'
		)])

		diff_acronyms = merged_ids.transform({
			'id_a': lambda col: col.map(confs_a).map(operator.attrgetter('acronym')).str.upper(),
			'id_b': lambda col: col.map(confs_b).map(operator.attrgetter('acronym')).str.upper(),
		}).query('id_a != id_b')

		print('Merges with differing acronyms:')
		if debug:
			# Print pair info and conf info for all candidates in merges with different acronyms
			if debug is True:
				debug_confs_a = merged_ids.loc[[*diff_acronyms.index], 'id_a'].to_list()
				debug_confs_b = merged_ids.loc[[*diff_acronyms.index], 'id_b'].to_list()
			else:
				debug_confs_a = idx_a.loc[idx_a.index.intersection(debug)].to_list()
				debug_confs_b = idx_b.loc[idx_b.index.intersection(debug)].to_list()

			relevant_pairs = all_compared_pairs.query(f"id_a in {debug_confs_a} or id_b in {debug_confs_b}")
			detailed_scores = relevant_pairs.apply(
				lambda row: pd.Series(ConfMetaData._difference(confs_a[row['id_a']], confs_b[row['id_b']])),
				axis='columns'
			).rename(columns=dict(enumerate(['acronym', 'type', 'org', 'topic', 'qualif', 'num'])))
			acronyms = relevant_pairs[['id_a', 'id_b']].transform({
				'id_a': lambda col: col.map(confs_a).map(operator.attrgetter('acronym')).str.upper(),
				'id_b': lambda col: col.map(confs_b).map(operator.attrgetter('acronym')).str.upper(),
			}).rename(columns=lambda name: f'acronym_{name[-1]}').sort_values(['acronym_a', 'acronym_b'])
			selected = relevant_pairs.index.to_series(name='merged').isin(merged_ids.index).map({True: '*', False: ''})

			if not len(debug_confs_a) and not len(debug_confs_b):
				print('None\n')
				return merged

			print()
			print(acronyms.join([relevant_pairs, selected, detailed_scores]))
			print()
			print('\nconfs_a:')
			print(('- ' + confs_a[debug_confs_a].map(str)).str.cat(sep='\n'))
			print('\nconfs_b:')
			print(('- ' + confs_b[debug_confs_b].map(str)).str.cat(sep='\n'))
			print()
		else:
			# Print pair info for all merged conferences with different acronyms
			full_scores = merged_ids.loc[diff_acronyms.index].apply(
				lambda row: pd.Series(ConfMetaData._difference(confs_a[row['id_a']], confs_b[row['id_b']])),
				axis='columns'
			).rename(columns=dict(enumerate(['acronym', 'type', 'org', 'topic', 'qualif', 'num'])))

			if not diff_acronyms.size:
				print('None\n')
				return merged

			print(diff_acronyms.join(full_scores))
			print()


		return merged


class GGSRanking(Ranking):
	_url_ggsrank = 'https://scie.lcc.uma.es/gii-grin-scie-rating/conferenceRating.jsf'
	_file = 'ggs.csv'
	_col_order = ['acronym', 'title', 'rank']


	@classmethod
	def _add_implicit_columns(cls, confs: pd.DataFrame) -> pd.DataFrame:
		""" Data that is not saved in the csv """
		confs['ranksys'] = 'GGS2021'
		confs['field'] = None
		return confs


	@classmethod
	def _load_confs(cls) -> pd.DataFrame:
		""" Load conferences from a file where we have the values cached cleanly.  """
		return super()._load_confs().pipe(cls._add_implicit_columns)


	@classmethod
	def _fetch_confs(cls) -> pd.DataFrame:
		""" Fetch unparsed conference info from the GGS website """
		soup = RequestWrapper.get_soup(cls._url_ggsrank, 'cache/gii-grin-scie-rating_conferenceRating.html')
		link = cast(bs4.Tag, soup.find('a', attrs={'href': lambda url: url.split(';jsessionid=')[0].endswith('.xlsx')}))
		file_url = parse.urljoin(cls._url_ggsrank, link.attrs['href'])

		df = pd.read_excel(file_url, header=1, usecols=['Title', 'Acronym', 'GGS Rating'])\
			   .rename(columns={'GGS Rating': 'rank', 'Title': 'title', 'Acronym': 'acronym'})

		# Remove sponsor from acronym
		df['acronym'] = df['acronym'].str.replace(r'^(IEEE|ACM)[-_/ ]', '', regex=True)

		# Drop old stuff or no acronyms (as they are used for lookup)
		df = df[~(df['rank'].str.contains('discontinued|now published as journal', case=False) |
				  df['acronym'].isna() | df['acronym'].str.len().eq(0))]

		ok_rank = df['rank'].str.match('^[A-Z][+-]*$')
		print('Non-standard ratings:')
		print(df['rank'].mask(ok_rank).value_counts().to_string())
		df['rank'] = df['rank'].where(ok_rank)
		df['title'] = df['title'].str.replace(';', ',').str.title()\
				.str.replace(r'\b(Acm|Ieee)\b', lambda m: m[1].upper(), regex=True)\
				.str.replace(r'\b(On|And|In|Of|For|The|To|Its)\b', lambda m: m[1].lower(), regex=True)

		# A few ad-hoc fixes
		# 1) Wrong language for our matching system
		df.loc[df['acronym'].eq('CLEI') & df['title'].eq('Conferencia Latinoamericana De Informática'),
			   'title'] = 'Latin American Conference on Informatics'

		# 2) Renamed from workshop to conference
		df.loc[df['acronym'].eq('EUMAS') & df['title'].eq('European Workshop on Multi-Agent Systems'),
			   'title'] = 'European Conference on Multi-Agent Systems'

		# 3) Alternate acronyms suffixed _A or -AUS (and 1 prefixed AUS-) mean worse results in wikicfp search
		df.loc[df['acronym'].eq('AUS-AI') & df['title'].eq('Australian Joint Conference on Artificial Intelligence'),
			   'acronym'] = 'AJCAI'
		df['acronym'] = df['acronym'].str.replace('(_A|-AUS)$', '', regex=True)

		def sort_ranks(series):
			return series.map(Conference._ranks).fillna(len(Conference._ranks)) if series.name == 'rank' else series

		return df[cls._col_order].sort_values(by=cls._col_order, key=sort_ranks).pipe(cls._add_implicit_columns)


class CoreRanking(Ranking):
	""" Utility class to scrape CORE conference listings and generate `~Conference` objects.  """
	_file = 'core.csv'
	_col_order = ['acronym', 'title', 'ranksys', 'rank', 'field']
	_url_corerank = 'http://portal.core.edu.au/conf-ranks/?search=&by=all&source={}&sort=arank&page={}'
	_source = 'CORE2023'
	_for_file = 'for_codes.json'


	@classmethod
	def _fetch_confs(cls) -> pd.DataFrame:
		""" Fetch unparsed conference info from the core website """
		# fetch page 1 outside loop to get page/result counts, will be in cache for loop access
		soup = RequestWrapper.get_soup(cls._url_corerank.format(cls._source, 1), 'cache/ranked_{1}.html')

		result_count_re = re.compile('Showing results 1 - ([0-9]+) of ([0-9]+)')
		result_count = cast(bs4.NavigableString, soup.find(string=result_count_re))
		per_page, n_results = map(int, cast(Match, result_count_re.search(result_count)).groups())
		pages = (n_results + per_page - 1) // per_page

		cfp_data = []
		with click.progressbar(label='fetching CORE list…', length=n_results) as prog:
			for p in range(1, pages + 1):
				soup = RequestWrapper.get_soup(cls._url_corerank.format(cls._source, p), f'cache/ranked_{p}.html')

				table = cast(bs4.Tag, soup.find('table'))
				rows = cast(Iterator[bs4.Tag], iter(table.find_all('tr')))

				headers = [' '.join(r.text.split()).lower() for r in next(rows).find_all('th')]

				tpos = headers.index('title')
				apos = headers.index('acronym')
				rpos = headers.index('rank')
				fpos = headers.index('primary for')

				for row in rows:
					val = [' '.join(r.text.split()) for r in row.find_all('td')]
					cfp_data.append([val[apos], val[tpos], val[rpos], val[fpos]])
					prog.update(1)

		cfps = pd.DataFrame(cfp_data, columns=['acronym', 'title', 'rank', 'field'])
		cfps.insert(3, 'ranksys', cls._source)

		# Manually add some missing conferences from previous year data.
		cfps = pd.concat(ignore_index=True, objs=[cfps, pd.DataFrame(columns=cfps.columns, data=[
			('MICRO',		'International Symposium on Microarchitecture',					'A',  'CORE2018', '4601'),
			('VLSI',		'Symposia on VLSI Technology and Circuits',						'A',  'CORE2018', '4009'),
			('ICC',			'IEEE International Conference on Communications',				'B',  'CORE2018', '4006'),
			('IEEE RFID',	'IEEE International Conference on Radio Frequency '
							'Identification',												'B',  'CORE2018', '4006'),
			('M2VIP',		'Mechatronics and Machine Vision in Practice',					'B',  'CORE2018', '4611'),
			('ICASSP',		'IEEE International Conference on Acoustics, Speech '
							'and Signal Processing',										'B',  'CORE2018', '4006'),
			('RSS',			'Robotics: Science and Systems',								'A*', 'CORE2018', '4611'),
			('BuildSys',	'ACM International Conference on Systems for Energy-Efficient '
							'Built Environments',											'A',  'CORE2018', '4606'),
			('DAC',			'Design Automation Conference',									'A',  'CORE2018', '4606'),
			('FSR',			'International Conference on Field and Service Robotics',		'A',  'CORE2018', '4602'),
			('CDC',			'IEEE Conference on Decision and Control',						'A',  'CORE2018', '4009'),
			('ASAP',		'International Conference on Application-specific Systems, '
							'Architectures and Processors',									'A',  'CORE2018', '4606'),
			('ISR',			'International Symposium on Robotics',							'A',  'CORE2018', '4007'),
			('ISSCC',		'IEEE International Solid-State Circuits Conference',			'A',  'CORE2018', '4009'),

			('RANDOM',		'International Workshop on Randomization and Computation',		'A',  'CORE2021', '4613'),
			('SIMULTECH',	'International Conference on Simulation and '
							'Modeling Methodologies, Technologies and Applications',		'C',  'CORE2021', '4606'),
			('ICCS',		'International Conference on Conceptual Structures',			'B',  'CORE2021', '4613'),
		])])

		cfps['title'] = cfps['title'].transform(cls.strip_trailing_paren)

		cfps['acronym'] = cfps['acronym'].str.replace(r'^(IEEE|ACM)[-_/ ]', '', regex=True)

		with open(cls._for_file, 'r') as f:
			forcodes = json.load(f)

		cfps['field'] = cfps['field'].map(forcodes)

		text_ranks = cfps['rank']
		non_standard_ranks = cfps['rank'][text_ranks.isna() & cfps['rank'].notna()].value_counts()
		cfps['rank'] = text_ranks

		# Also normalize rankings
		local_ranks = cfps['rank'].str.startswith('National') | cfps['rank'].str.startswith('Regional')
		local_places = cfps['rank'].where(local_ranks).str[8:].str.strip('(): -').where(lambda s: s.str.len().gt(0))
		local_places = local_places.str.title().replace({'Usa': 'USA', 'S. korea': 'Korea'})

		cfps.loc[local_ranks, 'rank'] = cfps['rank'][local_ranks].str[:8]
		cfps.loc[local_places.notna(), 'rank'] += ': ' + local_places.dropna()

		standard_ranks = local_ranks | cfps['rank'].str.match(r'^(Australasian )?[A-Z]\*?$')
		non_standard_ranks = cfps['rank'][~standard_ranks].value_counts()
		cfps['rank'] = cfps['rank'].where(standard_ranks)

		if len(non_standard_ranks):
			print('Non-standard ratings:')
			width = max(map(len, non_standard_ranks.keys())) + 3
			for key, num in non_standard_ranks.items():
				print(f'{key:{width}} {num}')

		return cfps.sort_values(by=[*cfps.columns]).drop_duplicates()


def json_encode_dates(obj: datetime.date):
	if isinstance(obj, datetime.date):
		return obj.strftime(r'%Y%m%d')
	else:
		raise TypeError('{} not encodable'.format(obj))


@click.group(invoke_without_command=True, chain=True)
@click.option('--cache/--no-cache', default=True, help='Cache files in ./cache')
@click.option('--delay', type=float, default=0, help='Delay between requests to the same domain')
@click.option('--report-spelling/--no-report-spelling', default=True,
			  help='Whether to print a report on miss-spelled words')
@click.pass_context
def update(ctx: click.Context, cache: bool, delay: float, report_spelling: bool):
	""" Update the Core-CFP data. If no command is provided, update_confs is run.  """
	RequestWrapper.set_delay(delay)
	RequestWrapper.set_use_cache(cache)

	if not ctx.invoked_subcommand:
		# Default is to_update calls for papers
		cfps()


@update.result_callback()
def process_result(*args, **kwargs):
	print(f'Encountered {len(ConfMetaData._misspelled)} unrecognized miss-spelled words')
	if kwargs.get('report_spelling') and ConfMetaData._misspelled:
		ninfo = pd.Series(ConfMetaData._misspelled).str.len()
		print(ninfo.sort_values(ascending=False).map(lambda n: f'×{n}' if n > 1 else '').to_string())


@update.command()
def core():
	""" Update the cached list of CORE conferences """
	CoreRanking.update_confs()


@update.command()
def ggs():
	""" Update the cached list of GII-GRIN-SCIE (GGS) conferences """
	GGSRanking.update_confs()


@update.command(hidden=True)
@click.option('--debug/--no-debug', default=False,
			  help='Show debug output for differing acronyms (if no acronyms are selected)')
@click.option('--debug-acronym', default=[], multiple=True, help='Select debug output for selected acronyms')
def load_confs(debug: bool = False, debug_acronym: list[str] = []):
	""" Load and merge the the conference lists. """
	confs = Ranking.merge(CoreRanking.get_confs(), GGSRanking.get_confs(), debug=debug_acronym or debug).sort_values()

	if debug_acronym:
		show = confs.map(operator.attrgetter('acronym')).str.upper().str.match('|'.join(debug_acronym))
		print('\nResulting confs:')
		print(('- ' + confs[show].map(str)).str.cat(sep='\n'))


@update.command()
@click.option('--out', 'out_file', default='cfp.json', help='Output file for CFPs', type=click.Path(dir_okay=False))
@click.option('--debug/--no-debug', default=False, help='Show debug output')
def cfps(out_file: str, debug: bool = False):
	""" Update the calls for papers from the conference lists  """
	today = datetime.datetime.now().date()
	# use years from 6 months ago until next year
	search_years = range((today - datetime.timedelta(days=183)).year, (today + datetime.timedelta(days=365)).year + 1)

	confs = Ranking.merge(CoreRanking.get_confs(), GGSRanking.get_confs(), debug=debug).sort_values()

	def prog_show_conf(arg: tuple[int, Conference] | None, width: int = _term_columns - 50 - 36) -> str:
		if arg is None:
			return ''
		info = f'{arg[1].acronym} {arg[1].title}'
		return f'{info[:width - 3]}...' if len(info) > width else info

	progressbar = click.progressbar(confs.items(), label='fetching calls for papers…', width=36,
									item_show_func=prog_show_conf, length=len(confs),
									update_min_steps=len(confs) // 1000 if not RequestWrapper.delay else 1)

	conf_matching = []
	with progressbar as conf_iterator:
		for conf_id, conf in conf_iterator:
			for year in search_years:
				if debug:
					clean_print(f'\nLooking up CFP {conf} {year}')
				try:
					cfp, cmp, miss = WikicfpCFP.get_cfp(conf, year, debug=debug)

					assert cfp.url_cfp is not None, 'By definition of a fetched CFP'

				except CFPNotFoundError as err:
					if debug:
						print(f'> {err}')

				else:
					if debug:
						print('> Found')

					conf_matching.append((conf_id, cfp.id, year, cmp, miss))
					continue

				# possibly try other CFP providers?

				if year < today.year:
					continue

				# Use a fallback into which we can extrapolate
				if debug:
					print(f'> Adding empty cfp')

				cfp = CallForPapers.build(conf.acronym, year)
				print(year, cfp)
				conf_matching.append((conf_id, cfp.id, year, 999, len(cfp.__slots__)))

	with open('parsing_errors.txt', 'w') as errlog:
		print(*CallForPapers._errors, sep='\n', file=errlog)

	conf_matching_df = pd.DataFrame(
		conf_matching, columns=['conf_id', 'cfp_id', 'year', 'score', 'missing']
	).set_index(['conf_id', 'year'])

	# In some cases we have 2 related conferences that are thus close in terms of acronym, description, etc.
	# E.g. “INFOCOM” and “INFOCOM WKSHPS“ or “USENIX ATC” and “USENIX-STX“ so ensure we only output each cfp once.
	conf_matching_df = conf_matching_df.loc[conf_matching_df.groupby(['cfp_id'])['score'].idxmin()]

	# Don’t output conferences if all (remaining) cfps are fallback (≥ 8 missing infos)
	conf_matching_df = conf_matching_df[~conf_matching_df['missing'].ge(8).groupby(level='conf_id').transform('all')]

	def extrapolate(cfps: pd.Series[CallForPapers], n: int) -> pd.Series[CallForPapers]:
		prev_cfps = cfps.groupby(level=['conf_id']).shift(periods=n)
		return cfps.combine(prev_cfps, CallForPapers.extrapolate_missing)

	# Complete missing cfp info with previous iterations
	cfps = CallForPapers.all_built_cfps()
	full_cfps = conf_matching_df['cfp_id'].sort_index().map(cfps).pipe(extrapolate, n=1).pipe(extrapolate, n=2)

	# Convert all cfps / confs to lists of data to be written out
	cfp_data = full_cfps.map(CallForPapers.values).unstack('year', fill_value=[None] * len(CallForPapers.columns()))
	conf_data = cfp_data.index.to_series(name='conf').map(confs.to_dict()).map(Conference.values).map(list)

	# Combine all conference / cfp data and sort based on acronym
	out_years = [year for year in search_years if year >= today.year]
	all_data = conf_data.add(cfp_data[out_years].agg(list, axis='columns')).reindex_like(conf_data.str[0].sort_values())

	try:
		min_ctime = min(os.path.getctime(f) for f in glob.glob('cache/cfp_*.html'))
	except ValueError:
		scrape_date = datetime.datetime.now()
	else:
		scrape_date = datetime.datetime.fromtimestamp(min_ctime)

	with open(out_file, 'w') as out:
		print(f'{{"years": {json.dumps(out_years)}, "columns":\n{json.dumps(Conference.columns())},', file=out)
		print(f'"cfp_columns":\n{json.dumps(CallForPapers.columns())},', file=out)
		print('"data": [', file=out)

		to_string = functools.partial(json.dumps, default=json_encode_dates)
		print(all_data.map(to_string).str.cat(sep=',\n'), file=out)

		print(f'], "date": "{scrape_date.strftime("%Y-%m-%d")}"}}', file=out)


if __name__ == '__main__':
	update()
