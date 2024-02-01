const ranks = ['E', 'D', 'C', 'B-', 'B', 'A-', 'A', 'A+', 'A*', 'A++'];
// Indexes in a row of data
let confIdx = 0, titleIdx = 1, rankIdx = 2, rankingIdx = 3, fieldIdx = 4, cfpIdx = 5;
// Indexes in a cfp list
let abstIdx = 0, subIdx = 1, notifIdx = 2, camIdx = 3, startIdx = 4, endIdx = 5, origOffset = 6;
	linkIdx = 12, cfpLinkIdx = 13;

const today = new Date();

const month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const timeline_zero = Date.UTC(today.getFullYear(), today.getMonth() - 6, 1);
const date_zero = new Date(timeline_zero);

// some global variables
let timeline_max = Date.UTC(today.getFullYear(), today.getMonth() + 18, 0);
let date_max = new Date(timeline_max);
// % per month: 50px / duration of 1 month
let timeline_scale = 100 / (timeline_max - timeline_zero);

const timeline = document.getElementById('timeline');
const suggestions = document.querySelector('#suggestions');
const form = document.querySelector('form');
const filters = {};
let data = [];

let filtered_confs; // #search p.filter_conf

// the value we push into the hash
let sethash = '';

// Escape data to pass as regexp
RegExp.escape = s => s.replace(/[\\^$*+?.()|[\]{}]/g, '\\$&')

// timeout id to delay search update
let updateSearchTimeoutId = 0;


// Template elements that we can clone()
const marker = document.createElement('sup');
marker.className = 'est';
marker.textContent = '†';

const line = document.createElement('p');
line.appendChild(document.createElement('span')).className = 'acronyms';
line.appendChild(document.createElement('span')).className = 'timeblocks';
line.style.display = 'none';

const wikicfp = document.createElement('a').appendChild(document.createElement('img')).parentNode;
wikicfp.firstChild.src = 'wikicfplogo.png';
wikicfp.firstChild.alt = 'Wiki CFP logo';
wikicfp.firstChild.className += 'cfpurl';

const suggestion = document.createElement('li');
suggestion.appendChild(document.createElement('span')).className += 'conf';
suggestion.appendChild(document.createElement('span')).className += 'rank';
suggestion.appendChild(document.createElement('span')).className += 'field';
suggestion.appendChild(document.createElement('span')).className += 'title';
suggestion.style.display = 'none';

// Formatter for dates
const dateFormat = new Intl.DateTimeFormat('en', {
	weekday: 'short',
	year: 'numeric',
	month: 'short',
	day: 'numeric',
});

function ranksort(a, b)
{
	const rank_a = ranks.indexOf(a), rank_b = ranks.indexOf(b);
	// compare using positions
	if (rank_a >= 0 && rank_b >= 0)
		return rank_b - rank_a;
	// compare as strings
	else if(rank_a < 0 && rank_b < 0)
		return a > b;
	// return 1 for the element not negative
	else
		return rank_a < 0 ? 1 : -1;
}

function parseFragment()
{
	const hash_parts = window.location.hash.substr(1).split('&');
	let anchorYOffset = undefined;

	const result = hash_parts.reduce(function (result, item)
	{
		const parts = item.split('=', 2);

		if (parts.length > 1)
		{
			if (!result[parts[0]])
				result[parts[0]] = [];

			result[parts[0]].push(decodeURIComponent(parts[1]));
		}
		else if (item && document.getElementById(item))
			anchorYOffset = window.pageYOffset + document.getElementById(item).getBoundingClientRect().top;

		return result;
	}, {});

	if (result.length && anchorYOffset !== undefined)
		window.scroll(window.pageXOffset, anchorYOffset);

	return result;
}

function updateFragment()
{
	const params = Array.from(form.querySelectorAll('select')).reduce((params, sel) =>
		params.concat(Array.from(sel.selectedOptions).map(opt => `${sel.name}=${encodeURIComponent(opt.value)}`))
	, []).sort().filter((it, pos, arr) => pos === 0 || it !== arr[pos - 1]);

	/* get last part of &-separated fragment that contains no '=' */
	const anchor = window.location.hash.substr(1).split('&').reduce(function (prev, item)
	{
		return item.indexOf('=') < 0 ? item : prev;
	}, null);

	if (anchor)
		params.push(anchor);

	sethash = '#' + params.join('&');
	if (window.location.hash !== sethash)
		window.location.hash = sethash;
}

function makeTimelineLegend()
{
	const box = document.getElementById('timeline_header');
	while (box.hasChildNodes())
		box.firstChild.remove();

	const months = document.createElement('p');
	months.id = 'months';
	months.appendChild(document.createElement('span')).className += 'acronyms';
	months.appendChild(document.createElement('span')).className += 'timeblocks';

	const year_from = date_zero.getFullYear(), year_diff = date_max.getFullYear() - year_from;

	for (let m = date_zero.getMonth(); m <= date_max.getMonth() + 12 * year_diff; m++)
	{
		const from = Date.UTC(year_from, m, 1);
		const until = Date.UTC(year_from, m + 1, 0);

		const month = months.lastChild.appendChild(document.createElement('span'));
		month.textContent = month_name[m % 12];
		month.style.width = `${(until - from) * timeline_scale}%`
		month.style.left = `${(from - date_zero) * timeline_scale}%`
		if (m % 12 === 0)
			month.className += 'first';
	}
	months.lastChild.firstChild.className += 'first';

	const years = document.createElement('p');
	years.id = 'years';
	years.appendChild(document.createElement('span')).className += 'acronyms';
	years.appendChild(document.createElement('span')).className += 'timeblocks';

	for (let y = year_from; y <= year_from + year_diff; y++)
	{
		const from = Math.max(date_zero, Date.UTC(y, 0, 1));
		const until = Math.min(date_max, Date.UTC(y + 1, 0, 0));

		const year = years.lastChild.appendChild(document.createElement('span'));
		year.textContent = y;
		year.style.width = `calc(${(until - from) * timeline_scale}% - 1px)`;
		year.style.left = `${(from - date_zero) * timeline_scale}%`;
	}

	const now = document.createElement('p');
	now.id = 'now';
	now.appendChild(document.createElement('span')).className += 'acronyms';
	now.appendChild(document.createElement('span')).className += 'timeblocks';

	const day = now.lastChild.appendChild(document.createElement('span'));
	day.className += 'today';
	day.style.left = `${(today.valueOf() - date_zero) * timeline_scale}%`;

	box.appendChild(years);
	box.appendChild(months);
	box.appendChild(now);
}

function parseDate(str)
{
	if (!str)
		return null;

	const date = Date.UTC(str.substring(0, 4), str.substring(4, 6) - 1, str.substring(6, 8));
	return Math.min(timeline_max, Math.max(timeline_zero, date));
}

function makeTimelineDuration(cls, from, until, tooltip_text, from_orig, until_orig)
{
	const span = document.createElement('span');
	span.className = cls;
	span.style.width = (until - from) * timeline_scale + '%';
	span.style.left = (from - timeline_zero) * timeline_scale + '%';

	const tooltip = span.appendChild(document.createElement('span'));
	tooltip.className = 'tooltip';
	tooltip.innerHTML = tooltip_text; /* to put html tags in tooltip_text */

	if ((from + until) < (timeline_zero + timeline_max))
		tooltip.style.left = '0';
	else
		tooltip.style.right = '0';

	if (from_orig === false)
		span.appendChild(marker.cloneNode(true)).style.left = '0';
	if (until_orig === false)
		span.appendChild(marker.cloneNode(true)).style.left = '100%';

	return span;
}


function makeTimelinePunctual(cls, date, content, tooltip_text, date_orig)
{
	const span = document.createElement('span');
	span.className = cls;
	span.style.width = '.5em';
	span.innerHTML = content;
	span.style.left = 'calc(' + (date - timeline_zero) * timeline_scale + '% - .25em)';

	const tooltip = span.appendChild(document.createElement('span'));
	tooltip.className = 'tooltip';
	tooltip.innerHTML = tooltip_text; /* to put html tags in tooltip_text */

	if (date < (timeline_zero + timeline_max) / 2)
		tooltip.style.left = '0';
	else
		tooltip.style.right = '0';

	if (date_orig === false)
		span.appendChild(marker.cloneNode(true)).style.left = '.75em';

	return span;
}

function est(isOrig) {
	return isOrig === false ? 'Estimated ' : '';
}

function objMap(obj, func) {
	return Object.fromEntries(
		Object.entries(obj).map(([key, val], idx) => [key, func(val, key, idx)])
	)
}

function makeTimelineItem(row)
{
	const p = line.cloneNode(true);
	renderAcronym(p.firstChild, row);

	const dateIdx = {abst: abstIdx, sub: subIdx, notif: notifIdx, cam: camIdx, start: startIdx, end: endIdx};
	const blocks = p.lastChild;
	for (const cfp of row.slice(cfpIdx))
	{
		// get the row for this year, with friendly names
		const date = objMap(dateIdx, idx => parseDate(cfp[idx]));
		const orig = objMap(dateIdx, idx => cfp[idx + origOffset]);
		const text = objMap(date, dt => dt && dateFormat.format(dt));
		const acronym = `${row[confIdx]} ${(cfp[startIdx] || cfp[endIdx] || 'last').slice(0, 4)}`;

		if (!date.abst)
			date.abst = date.sub;
		else if (!date.sub)
			date.sub = date.abst;

		if (date.sub && date.notif && date.notif >= date.sub)
		{
			if (date.sub > date.abst)
			{
				const tooltip = `${est(orig.abst)}${acronym} registration ${text.abst}`;
				blocks.appendChild(makeTimelineDuration('abstract', date.abst, date.sub, tooltip, orig.abst));
			}

			const tooltip = [
				`${est(orig.sub)}${acronym} submission ${text.sub},`,
				`${est(orig.notif).toLowerCase()}notification ${text.notif}`,
			].join('<br />');
			blocks.appendChild(makeTimelineDuration('review', date.sub, date.notif, tooltip, orig.sub, orig.notif));
		}
		else if (date.sub)
		{
			const tooltip = `${est(orig.sub)}${acronym} submission ${text.sub}`;
			blocks.appendChild(makeTimelinePunctual('date.sub', date.sub, '<sup>◆</sup>', tooltip, orig.sub));
		}

		if (date.cam)
		{
			const tooltip = `${est(orig.cam)}${acronym} final version ${text.cam}`;
			blocks.appendChild(makeTimelinePunctual('date.cam', date.cam, '<sup>∎</sup>', tooltip, orig.cam));
		}

		if (date.start && date.end && date.end >= date.start)
		{
			const tooltip = `${acronym} ${est(orig.start && orig.end).toLowerCase()}from ${text.start} to ${text.end}`;
			blocks.appendChild(makeTimelineDuration('conf', date.start, date.end, tooltip, undefined, orig.end));
		}
	}

	return timeline.appendChild(p);
}


function makeSuggestionItem(row)
{
	const item = suggestion.cloneNode(true);

	item.children[0].textContent = row[confIdx];
	item.children[1].textContent = row[rankIdx].map(
		(val, idx) => `${val || 'unrated'} (${row[rankingIdx][idx]})`
	).join(', ');
	item.children[2].textContent = row[fieldIdx] == '(missing)' ? '': row[fieldIdx];
	item.children[3].textContent = row[titleIdx];

	const opt = Array.from(form.querySelector('select[name="conf"]').options).find(opt => opt.value === row[confIdx]);
	item.onclick = () =>
	{
		opt.selected = true;
		opt.parentNode.onchange();
		form.querySelector('input[name="search"]').value = '';
	}

	return suggestions.appendChild(item);
}


function makeSelectedItem(row)
{
	const item = document.createElement('span');
	item.textContent = row[confIdx];
	item.title = row[titleIdx];

	const opt = Array.from(form.querySelector('select[name="conf"]').options).find(opt => opt.value === row[confIdx]);
	item.onclick = () => { opt.selected = false; opt.parentNode.onchange(); }

	// insert at N-2
	filtered_confs.insertBefore(item, filtered_confs.lastChild.previousSibling);
}


function hideSuggestions()
{
	Array.from(suggestions.children).filter(conf => conf.style.display !== 'none')
									.forEach(conf => { conf.style.display = 'none' });
}


function delayedUpdateSearch(value)
{
	const terms = value.split(/[ ;:,.]/).filter(val => val && val.length >= 2);
	const search = terms.map(val => new RegExp(RegExp.escape(val), 'iu'));

	hideSuggestions();

	// -> all(words) -> any(columns)
	if (search.length)
		data.forEach((row, idx) =>
		{
			if (search.every(r => r.test(row[confIdx]) || r.test(row[titleIdx])))
				suggestions.children[idx].style.display = 'block';
		});

	updateSearchTimeoutId = 0;
}

function updateSearch()
{
	if (updateSearchTimeoutId)
		clearTimeout(updateSearchTimeoutId);

	updateSearchTimeoutId = setTimeout(delayedUpdateSearch, 150, this.value)
}


function setColumnFilter(select, col_id)
{
	const val = Array.from(select.selectedOptions).map(opt => RegExp.escape(opt.value));
	const regex = val.length ? `^(${val.join('|')})$` : '';

	if (regex)
		filters[col_id] = new RegExp(regex);
	else
		delete filters[col_id];
}

// this is the select
function updateFilter()
{
	const column_id = this.getAttribute('column_id');
	setColumnFilter(this, column_id);

	filterUpdated().then(updateFragment);
}

async function filterUpdated(search)
{
	Array.from(timeline.children).filter(conf => conf.style.display !== 'none').forEach(conf => {
		conf.style.display = 'none'
	});
	Array.from(filtered_confs.children).slice(0, timeline.children.length)
		.filter(conf => conf.style.display !== 'none').forEach(conf => conf.style.display = 'none');

	// Every filter needs to match at least one of its values
	data.map((row, idx) =>
		Object.entries(filters).reduce((ret, [col, regex]) => Object.assign(ret,
			{[col]: Array.isArray(row[col]) ? row[col].some(entry => regex.test(entry)) : regex.test(row[col])}
		), {index: idx}
	)).forEach(({ index, ...row_filters }) =>
	{
		const show = Object.values(row_filters).every(val => val);
		const tl_display = show ? 'block' : 'none';
		const conf_display = row_filters[confIdx] === true ? 'inline-block' : 'none';

		if (timeline.children[index].style.display !== tl_display)
			timeline.children[index].style.display = tl_display;

		if (filtered_confs.children[index].style.display !== conf_display)
			filtered_confs.children[index].style.display = conf_display;

		if (filtered_confs.children[index]) {
			const conf_class = show ? '' : 'filtered-out';
			if (filtered_confs.children[index].className !== conf_class)
				filtered_confs.children[index].className = conf_class;
		}
	});
}

function makeFilter(colIdx, name, sortfunction)
{
	let values = data.map(row => row[colIdx]);
	if (name === 'rank')
		values = [].concat(...values).map(rank => rank || '(unranked)');
	values = values.sort(sortfunction).filter((val, idx, arr) => idx === 0 || val !== arr[idx - 1]);

	const p = document.createElement('p');
	p.className += 'filter_' + name

	const select = p.appendChild(document.createElement('select'));
	select.multiple = true;
	select.name = name;
	select.size = values.length;
	select.setAttribute('column_id', colIdx);

	const clear = p.appendChild(document.createElement('button'));
	clear.textContent = 'clear';

	values.forEach(t =>
	{
		const option = select.appendChild(document.createElement('option'));
		option.textContent = t;
		option.value = t === '(unranked)' ? null : t;
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
	const selects = Array.from(form.querySelectorAll('select'));
	const selectedValues = parseFragment();

	selects.forEach(sel =>
	{
		sel.selectedIndex = -1;
		const values = selectedValues[sel.name] || (sel.name == 'scope' ? ['0'] : []);
		if (!values.length)
			return;
		Array.from(sel.options).forEach(opt => { opt.selected = values.indexOf(opt.value) >= 0 });
	});

	selects.forEach(sel => sel.onchange());
}

function renderAcronym(p, row)
{
	let conf = document.createElement('span');

	for (const cfp of row.slice(cfpIdx).toReversed())
		if (cfp[linkIdx] && cfp[linkIdx] != '(missing)')
		{
			conf = document.createElement('a');
			conf.href = cfp[linkIdx]
			break;
		}

	conf.textContent = row[confIdx];
	conf.title = row[titleIdx];

	p.appendChild(conf);
	p.innerHTML += '&nbsp;'

	let rating = document.createElement('span');
	rating.textContent = row[rankIdx].filter(rank => rank).join(',');
	rating.title = row[rankIdx].map((val, idx) => `${val || 'unrated'} (${row[rankingIdx][idx]})`).join(', ');
	rating.className = 'ratingsys';

	p.appendChild(rating);
	p.innerHTML += '&nbsp;'

	for (const cfp of row.slice(cfpIdx).toReversed())
		if (cfp[cfpLinkIdx] && cfp[cfpLinkIdx] != '(missing)')
		{
			const cfpLink = p.appendChild(wikicfp.cloneNode(true));
			cfpLink.href = cfp[cfpLinkIdx];
			cfpLink.title = `Latest ${row[confIdx]} CFP on WikiCFP`;
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
	// First update global variables from fetched data
	data = json['data'];

	confIdx    = json['columns'].indexOf('Acronym')
	titleIdx   = json['columns'].indexOf('Title')
	rankingIdx = json['columns'].indexOf('Rank system')
	rankIdx    = json['columns'].indexOf('Rank')
	fieldIdx   = json['columns'].indexOf('Field')
	cfpIdx     = json['columns'].length

	abstIdx    = json['cfp_columns'].indexOf('Abstract Registration Due')
	subIdx     = json['cfp_columns'].indexOf('Submission Deadline')
	notifIdx   = json['cfp_columns'].indexOf('Notification Due')
	camIdx     = json['cfp_columns'].indexOf('Final Version Due')
	startIdx   = json['cfp_columns'].indexOf('startDate')
	endIdx     = json['cfp_columns'].indexOf('endDate')
	linkIdx    = json['cfp_columns'].indexOf('Link')
	cfpLinkIdx = json['cfp_columns'].indexOf('CFP url')

	origOffset = json['cfp_columns'].indexOf('orig_abstract')

	// Use lexicographic sort for dates, in format YYYYMMDD. NB: some dates are null.
	const mindate = [
		date_zero.getFullYear(),
		date_zero.getMonth() + 1,
		date_zero.getDate(),
	].map(num => String(num).padStart(2, '0')).join('');

	const maxdate = data.reduce((curmax, row) => Math.max(
		curmax,
		...row.slice(cfpIdx).map(cfp => cfp[endIdx] || cfp[startIdx] || cfp[subIdx])
	), mindate);

	// get last day of month
	timeline_max = Date.UTC(Math.floor(maxdate / 10000), Math.floor(maxdate / 100) % 100, 0);
	timeline_scale = 100 / (timeline_max - timeline_zero);
	date_max = new Date(timeline_max);

	makeTimelineLegend();

	// sort the data per upcoming deadline date
	const sortIdx = [subIdx, abstIdx, startIdx, endIdx];
	const nowdate = [
		today.getFullYear(),
		today.getMonth() + 1,
		today.getDate(),
	].map(num => String(num).padStart(2, '0')).join('');
	const sortdates = data.map(row => row.slice(cfpIdx)
		// Find one non-null date per cfp using sortIdx preference
		.map(cfp => sortIdx.map(idx => cfp[idx]).find(date => date !== null))
		// Find the best cfp: first after today, or first if all are after today
		.map(cfpdate => [cfpdate <= nowdate, cfpdate]).sort()[0]
	)

	// Now sort resulting dates and apply sort to data
	data = sortdates.map((date, idx) => [date, idx]).sort().map(([date, idx]) => data[idx]);

	document.getElementById('head').appendChild(
		document.createTextNode(` The last scraping took place on ${json['date']}.`)
	);

	document.getElementById('search').appendChild(makeFilter(confIdx, "conf"));
	filtered_confs = form.querySelector('p.filter_conf');

	const filters = document.getElementById('filters');
	filters.appendChild(makeFilter(rankIdx, "rank", ranksort));
	filters.appendChild(makeFilter(fieldIdx, "field"));

	const search = form.querySelector('input[name="search"]');
	search.onkeypress = updateSearch
	search.onkeyup = updateSearch
	search.onfocus = updateSearch
	search.onblur = () => setTimeout(hideSuggestions, 100)

	data.forEach((row, idx) =>
	{
		makeTimelineItem(row);
		makeSuggestionItem(row);
		makeSelectedItem(row);
	});

	// Initial fragment
	filterFromFragment();

	window.addEventListener('hashchange', filterFromFragment);

	// add data to Timeline, but only filtered
	filterUpdated();

	document.getElementById('loading').style.display = 'none';
}

function parsingErrors(content)
{
	const table = document.getElementById('error_log');
	for (const error of content.split('\n'))
	{
		if (!error.trim().length)
			continue;

		const [conf, errmsg, url, fixed] = error.replace(/ -- /g, ' – ').split(';');
		const err = document.createElement('tr');
		const link = err.appendChild(document.createElement('td')).appendChild(document.createElement('a'));
		link.textContent = conf;
		link.href = url;
		err.appendChild(document.createElement('td')).textContent = errmsg;

		table.appendChild(err).className = fixed;
	}

	const label = document.querySelector('label[for=collapse_errors]');
	label.textContent = (table.children.length - 1) + ' ' + label.textContent;
}

function fetchData(page, result_handler)
{
	req = new XMLHttpRequest();
	req.overrideMimeType('application/json; charset="UTF-8"');
	req.addEventListener('load', evt => result_handler(evt.target.responseText));
	req.open('get', page);
	req.send();
}
