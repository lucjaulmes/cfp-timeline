const ranks = ['D', 'C', 'B', 'A', 'A*'];
const confIdx = 0, titleIdx = 1, rankIdx = 2, fieldIdx = 3, linkIdx = 16, cfpIdx = 17;
const abstIdx = 4, subIdx = 5, notifIdx = 6, camIdx = 7, startIdx = 8, endIdx = 9;
const yearIdx = 4, yearOffset = 14, origOffset = 6;
const today = new Date(), year = today.getFullYear();

const month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const timeline_zero = Date.UTC(today.getFullYear(), today.getMonth() - 6, 1);

// some global variables
var timeline_max = Date.UTC(today.getFullYear(), today.getMonth() + 18, 0);
// % per month: 50px / duration of 1 month
var timeline_scale = 100 / (timeline_max - timeline_zero);

var timeline = document.getElementById('timeline'), n_years = 1;
var timeline_conf_lookup = {};
var form = document.querySelector('form');
var data = [], filters = {}, columns = [];

// the value we push into the hash
var sethash = '';

// Escape data to pass as regexp
RegExp.escape = s => s.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')


/* Template elements that we can clone() */
var marker = document.createElement('sup');
marker.className = 'est';
marker.textContent = '†';

var line = document.createElement('p');
line.appendChild(document.createElement('span')).className = 'acronyms';
line.appendChild(document.createElement('span')).className = 'timeblocks';
line.style.display = 'none';

var wikicfp = document.createElement('img');
wikicfp.src = 'wikicfplogo.png';
wikicfp.alt = 'Wiki CFP logo';
wikicfp.className += 'cfpurl';
wikicfp = document.createElement('a').appendChild(wikicfp).parentNode;


function ranksort(a, b)
{
	var ra = ranks.indexOf(a), rb = ranks.indexOf(b);
	// compare using positions
	if (ra >= 0 && rb >= 0) return rb - ra;
	// compare as strings
	else if(ra < 0 && rb < 0) return a > b;
	// return 1 for the element not negative
	else
		return ra < 0 ? 1 : -1;
}

function parseFragment()
{
	var hash = window.location.hash.substr(1).split('&');
	var goto = undefined;

	var result = hash.reduce(function (result, item)
	{
		var parts = item.split('=', 2);

		if (parts.length > 1)
		{
			if (!result[parts[0]])
				result[parts[0]] = [];

			result[parts[0]].push(decodeURIComponent(parts[1]));
		}
		else if (item && document.getElementById(item))
			goto = window.pageYOffset + document.getElementById(item).getBoundingClientRect().top;

		return result;
	}, {});

	if (result.length && goto !== undefined)
		window.scroll(window.pageXOffset, goto);

	return result;
}

function updateFragment()
{
	var params = Array.from(form.querySelectorAll('select')).reduce(
		(params, sel) => params.concat(Array.from(sel.selectedOptions).map(opt => sel.name + '=' + opt.value))
	, []).sort().filter((it, pos, arr) => pos === 0 || it !== arr[pos - 1]);

	/* get last part of &-separated fragment that contains no '=' */
	var goto = window.location.hash.substr(1).split('&').reduce(function (prev, item)
	{
		return item.indexOf('=') < 0 ? item : prev;
	}, null);

	if (goto)
		params.push(goto);

	sethash = '#' + params.join('&');
	if (window.location.hash != sethash)
		window.location.hash = sethash;
}

function makeTimelineLegend()
{
	var box = document.getElementById('timeline_header');
	while (box.hasChildNodes())
		box.firstChild.remove();

	var startDate = new Date(timeline_zero),
		endDate = new Date(timeline_max);

	var months = document.createElement('p');
	months.id = 'months';
	months.appendChild(document.createElement('span')).className += 'acronyms';
	months.appendChild(document.createElement('span')).className += 'timeblocks';

	for (var m = startDate.getMonth(); m <= endDate.getMonth() + 12 * (endDate.getFullYear() - startDate.getFullYear()); m++)
	{
		var from = Date.UTC(startDate.getFullYear(), m, 1);
		var until = Date.UTC(startDate.getFullYear(), m + 1, 0);

		var month = months.lastChild.appendChild(document.createElement('span'));
		month.textContent = month_name[m % 12];
		month.style.width = (until - from) * timeline_scale + '%'
		month.style.left = (from - timeline_zero) * timeline_scale + '%'
		if (m == startDate.getMonth() || m % 12 == 0) month.className += 'first';
	}

	var years = document.createElement('p');
	years.id = 'years';
	years.appendChild(document.createElement('span')).className += 'acronyms';
	years.appendChild(document.createElement('span')).className += 'timeblocks';

	for (var y = startDate.getFullYear(); y <= endDate.getFullYear(); y++)
	{
		var from = Math.max(timeline_zero, Date.UTC(y, 0, 1));
		var until = Math.min(timeline_max, Date.UTC(y + 1, 0, 0));

		var year = years.lastChild.appendChild(document.createElement('span'));
		year.textContent = y;
		year.style.width = 'calc(' + (until - from) * timeline_scale + '% - 1px)';
		year.style.left = (from - timeline_zero) * timeline_scale + '%';
	}

	var now = document.createElement('p');
	now.id = 'now';
	now.appendChild(document.createElement('span')).className += 'acronyms';
	now.appendChild(document.createElement('span')).className += 'timeblocks';

	var day = now.lastChild.appendChild(document.createElement('span'));
	day.className += 'today';
	day.style.left = ((today.valueOf() - timeline_zero) * timeline_scale) + '%';

	box.appendChild(years);
	box.appendChild(months);
	box.appendChild(now);
}

function parse_date(str)
{
	if (!str) return null;

	tok = str.split('-');
	return Math.max(timeline_zero, Date.UTC(tok[0], tok[1] - 1, tok[2]));
}

function makeTimelineDuration(cls, from, until, tooltip_text, from_orig, until_orig)
{
	var span = document.createElement('span');
	span.className = cls;
	span.style.width = (until - from) * timeline_scale + '%';
	span.style.left = (from - timeline_zero) * timeline_scale + '%';

	var tooltip = span.appendChild(document.createElement('span'));
	tooltip.className = 'tooltip';
	tooltip.innerHTML = tooltip_text; /* to put html tags in tooltip_text */

	if ((from + until) < (timeline_zero + timeline_max))
		tooltip.style.left = '0';
	else
		tooltip.style.right = '0';

	if (from_orig === false) span.appendChild(marker.cloneNode(true)).style.left = '0';
	if (until_orig === false) span.appendChild(marker.cloneNode(true)).style.left = '100%';

	return span;
}


function makeTimelinePunctual(cls, date, content, tooltip_text, date_orig)
{
	var span = document.createElement('span');
	span.className = cls;
	span.style.width = '.5em';
	span.innerHTML = content;
	span.style.left = 'calc(' + (date - timeline_zero) * timeline_scale + '% - .25em)';

	var tooltip = span.appendChild(document.createElement('span'));
	tooltip.className = 'tooltip';
	tooltip.innerHTML = tooltip_text; /* to put html tags in tooltip_text */

	if (date < (timeline_zero + timeline_max) / 2)
		tooltip.style.left = '0';
	else
		tooltip.style.right = '0';

	if (date_orig === false) span.appendChild(marker.cloneNode(true)).style.left = '.75em';

	return span;
}


function makeTimelineItem(row)
{
	var p = line.cloneNode(true);
	renderAcronym(p.firstChild, row);

	var blocks = p.lastChild;
	for (var y = 0; y < n_years; y++)
	{
		// get the row for this year, with friendl names
		var acronym = row[confIdx] + ' ' + (year + y), tooltip;
		var [abst, sub, notif, cam, start, end] = row.slice(abstIdx + y * yearOffset, endIdx + y * yearOffset + 1).map(parse_date);
		var [abstText, subText, notifText, camText, startText, endText] = row.slice(abstIdx + y * yearOffset, endIdx + y * yearOffset + 1);
		var [abstOrig, subOrig, notifOrig, camOrig, startOrig, endOrig] = row.slice(abstIdx + y * yearOffset + origOffset, endIdx + y * yearOffset + origOffset + 1);

		if (!abst) abst = sub;
		else if (!sub) sub = abst;

		if (sub && notif && notif >= sub)
		{
			if (sub > abst)
			{
				tooltip = (abstOrig === false ? 'Estimated' : '') + acronym + ' registration ' + abstText;
				blocks.appendChild(makeTimelineDuration('abstract', abst, sub, tooltip, abstOrig));
			}

			tooltip = (subOrig === false ? 'Estimated ' : '') +  acronym + ' submission ' + subText +
				',<br />' + (notifOrig === false ? 'estimated ' : '') + 'notification ' + notifText

			blocks.appendChild(makeTimelineDuration('review', sub, notif, tooltip, subOrig, notifOrig));
		}
		else if (sub)
		{
			tooltip = (subOrig === false ? 'Estimated ' : '') + acronym + ' submission ' + subText;
			blocks.appendChild(makeTimelinePunctual('sub', sub, '<sup>◆</sup>', tooltip, subOrig));
		}

		if (cam)
		{
			tooltip = (camOrig === false ? 'Estimated ' : '' ) + acronym + ' final version ' + camText;
			blocks.appendChild(makeTimelinePunctual('cam', cam, '<sup>∎</sup>', tooltip, camOrig));
		}

		if (start && end && end >= start)
		{
			tooltip = acronym + (startOrig === false || endOrig === false ? ' estimated from ' : ' from ')
						+ startText + ' to ' + endText;
			blocks.appendChild(makeTimelineDuration('conf', start, end, tooltip, undefined, endOrig));
		}
	}

	return timeline.appendChild(p);
}


function searchWords()
{
	var search = form.querySelector('input[name="words"]').value;
	return search.split(/[ ;:,.]/).filter(val => val);
}

function setColumnSearch(select, col_id)
{
	var val = Array.from(select.selectedOptions).map(opt => RegExp.escape(opt.value));
	var regex = val.length ? ('^(' + val.join('|') + ')$') : '';

	if (form.querySelector('select[name="scope"]').value == col_id)
	{
		var search = searchWords();
		if (val.length && search.length)
			regex += '|';
		regex += search.map(val => RegExp.escape(val)).join('|');
	}

	if (regex) filters[col_id] = new RegExp(regex);
	else delete filters[col_id];
}

// this is the select
function updateFilter()
{
	var column_id = this.getAttribute('column_id');
	setColumnSearch(this, column_id);

	filterUpdated().then(updateFragment);
}

function updateSearch()
{
	var col = parseInt(form.querySelector('select[name="scope"]').value);

	setGlobalSearch(col);
	if (col >= 0)
		setColumnSearch(form.querySelector('select[column_id="' + col + '"]'), col);

	filterUpdated().then(updateFragment);
}

// this is select.name="scope"
function updateSearchScope()
{
	var col = parseInt(this.value);

	setGlobalSearch(col);
	Array.from(this.options).map(opt => parseInt(opt.value)).forEach(val =>
	{
		if (val >= 0)
			setColumnSearch(form.querySelector('select[column_id="' + val + '"]'), col);
	});

	filterUpdated().then(updateFragment);
}

async function filterUpdated(search)
{
	Array.from(timeline.children).filter(conf => conf.style.display !== 'none').forEach(conf => conf.style.display = 'none');

	var loading = document.getElementById('loading');
	loading.style.display = 'block';

	data.filter(row => Object.keys(filters).every(col => filters[col].test(row[col]))).forEach(row =>
	{
		timeline_conf_lookup[row[confIdx]].style.display = 'block';
	});

	loading.style.display = 'none';
}

function makeFilter(colIdx, name, sortfunction)
{
	var values = data.map(row => row[colIdx]).sort(sortfunction).filter((val, idx, arr) => idx === 0 || val !== arr[idx - 1]);

	var opt = form.querySelector('select[name="scope"]').appendChild(document.createElement('option'));
	opt.value = colIdx;

	var p = document.createElement('p');
	p.className += 'filter_' + name
	opt.textContent = p.textContent = columns[colIdx].title;

	var select = p.appendChild(document.createElement('select'));
	select.multiple = true;
	select.name = name;
	select.size = values.length;
	select.setAttribute('column_id', colIdx);

	var clear = p.appendChild(document.createElement('button'));
	clear.textContent = 'clear';

	values.forEach(t =>
	{
		var option = select.appendChild(document.createElement('option'));
		option.textContent = t;
		option.value = t;
	});

	select.onchange = updateFilter
	clear.onclick = () =>
	{
		select.selectedIndex = -1;
		delete filters[colIdx];
		updateFilter.call(select);
	};

	return p;
}

function filterFromFragment()
{
	var selects = Array.from(form.querySelectorAll('select'));
	var selectedValues = parseFragment();

	selects.forEach(sel =>
	{
		sel.selectedIndex = -1;
		var values = selectedValues[sel.name] || (sel.name == 'scope' ? ['0'] : []);
		if (!values.length) return;
		Array.from(sel.options).forEach(opt => { opt.selected = values.indexOf(opt.value) >= 0 });
	});

	selects.filter(sel => sel.name !== 'scope').forEach(sel => sel.onchange());
}

function renderAcronym(p, row)
{
	var conf = document.createTextNode(row[confIdx]);

	for (var y = n_years - 1; y >= 0; y--)
		if (row[linkIdx + y * yearOffset] && row[linkIdx + y * yearOffset] != '(missing)')
		{
			conf = document.createElement('a').appendChild(conf).parentNode;
			conf.href = row[linkIdx + y * yearOffset]
			break;
		}

	p.appendChild(conf);
	p.innerHTML += '&nbsp;'

	for (y = n_years - 1; y >= 0; y--)
		if (row[cfpIdx + y * yearOffset] && row[cfpIdx + y * yearOffset] != '(missing)')
		{
			cfp = p.appendChild(wikicfp.cloneNode(true));
			cfp.href = row[cfpIdx + y * yearOffset];
			cfp.title = row[confIdx] + ' CFP on WikiCFP';
			break;
		}

	return p;
}

function markExtrapolated(td, data, rowdata, row, col)
{
	if (data && rowdata[col + origOffset] === false)
		td.className += 'extrapolated';
}

function notNull(val, idx)
{
	return val != null
}

function populatePage(json)
{
	data = json['data'];
	columns = json['columns'];

	var datesIdx = [], origIdx = [], urlIdx = [];
	n_years = (json['columns'].length - yearIdx) / yearOffset

	for (var y = 0; y < n_years; y++)
	{
		for (var d of [abstIdx, subIdx, notifIdx, camIdx, startIdx, endIdx])
		{
			datesIdx.push(d + y * yearOffset);
			origIdx.push(d + origOffset + y * yearOffset);
		}

		for (var u of [linkIdx, cfpIdx])
			urlIdx.push(u + y * yearOffset)
	}

	// Use lexicographic sort with cast to numbers for dates, i.e. parseInt(YYYYMMDD). NB: some dates are null.
	var mindate = new Date(timeline_zero);
	mindate = (mindate.getUTCFullYear() * 100 + (mindate.getUTCMonth() + 1)) * 100 + mindate.getUTCDate();

	var maxdate = data.reduce((curmax, row) =>
		Math.max.apply(null, datesIdx.map(col => (row[col] || '0').replace(/-/g, '')).concat([curmax]))
	, mindate);

	// get last day of month
	timeline_max = Date.UTC(Math.floor(maxdate / 10000), Math.floor(maxdate / 100) % 100, 0);
	timeline_scale = 100 / (timeline_max - timeline_zero);

	makeTimelineLegend();

	// sort the data per date
	const sortIdx = [
		  subIdx + (n_years - 1) * yearOffset,
		 abstIdx + (n_years - 1) * yearOffset,
		startIdx + (n_years - 1) * yearOffset,
		  endIdx + (n_years - 1) * yearOffset
	]

	// get sort-column subtractions, return first non-zero, or zero
	data.sort((rowA, rowB) => sortIdx.map(col => (rowA[col] || '').replace(/-/g, '') - (rowB[col] || '').replace(/-/g, ''))
									.find(diff => diff !== 0) || 0);

	data.forEach((row, idx) => timeline_conf_lookup[row[confIdx]] = makeTimelineItem(row));

	document.getElementById('head').textContent += ' The last scraping took place on ' + json['date'] + '.';

	var filters = document.getElementById('filters');
	filters.appendChild(makeFilter(confIdx, "conf"));
	filters.appendChild(makeFilter(rankIdx, "core", ranksort));
	filters.appendChild(makeFilter(fieldIdx, "field"));
	filters.appendChild(makeFilter(titleIdx, "title"));

	form.querySelector('select[name="scope"]').onchange = updateSearchScope
	form.querySelector('input[name="words"]').onkeypress = updateSearch

	// Initial fragment
	filterFromFragment();

	window.addEventListener('hashchange', filterFromFragment);

	// add data to Timeline, but only filtered
	filterUpdated();

	document.getElementById('loading').style.display = 'none';
}
