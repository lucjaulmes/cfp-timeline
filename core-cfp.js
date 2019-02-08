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

var datatable, timeline = document.getElementById('timeline'), n_years = 1;
var timeline_dom_cache = {};
var form = document.querySelector('form');

// the value we push into the hash
var sethash = '';


function ranksort(a, b)
{
	return ranks.indexOf(b) - ranks.indexOf(a);
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

	box.appendChild(years);
	box.appendChild(months);
}

function parse_date(str)
{
	if (!str) return null;

	tok = str.split('-');
	return Math.max(timeline_zero, Date.UTC(tok[0], tok[1] - 1, tok[2]));
}

/* Two template elements that we can clone() */
var marker = document.createElement('sup');
marker.className = 'est';
marker.textContent = '†';

var line = document.createElement('p');
line.appendChild(document.createElement('span')).className = 'acronyms';
line.appendChild(document.createElement('span')).className = 'timeblocks';

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


async function makeTimelineItem(data)
{
	var p = line.cloneNode(true);
	p.firstChild.innerHTML = renderAcronym(data[confIdx], 'display', data);

	var blocks = p.lastChild;
	for (var y = 0; y < n_years; y++)
	{
		// get the data for this year, with friendl names
		var acronym = data[confIdx] + ' ' + (year + y), tooltip;
		var [abst, sub, notif, cam, start, end] = data.slice(abstIdx + y * yearOffset, endIdx + y * yearOffset + 1).map(parse_date);
		var [abstText, subText, notifText, camText, startText, endText] = data.slice(abstIdx + y * yearOffset, endIdx + y * yearOffset + 1);
		var [abstOrig, subOrig, notifOrig, camOrig, startOrig, endOrig] = data.slice(abstIdx + y * yearOffset + origOffset, endIdx + y * yearOffset + origOffset + 1);

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

	return p;
}


async function addToTimeline(n, data)
{
	var p = timeline_dom_cache[data[confIdx]];
	if (!p)
	{
		p = await makeTimelineItem(data);
		timeline_dom_cache[data[confIdx]] = p;
	}

	// ignore conferences for which we don't have a single date
	if (p.lastChild.hasChildNodes())
		timeline.appendChild(p);
}

function searchWords()
{
	var search = form.querySelector('input[name="words"]').value;
	return search.split(/[ ;:,.]/).filter(val => val);
}

function setColumnSearch(select, column)
{
	var val = Array.from(select.selectedOptions).map(opt => $.fn.dataTable.util.escapeRegex(opt.value));
	var regex = val.length ? ('^(' + val.join('|') + ')$') : '';

	if (form.querySelector('select[name="scope"]').value == select.getAttribute('column_id'))
	{
		var search = searchWords();
		if (val.length && search.length)
			regex += '|';
		regex += search.map(val => $.fn.dataTable.util.escapeRegex(val)).join('|');
	}

	column.search(regex, true, false);
}

function setGlobalSearch(col)
{
	if (col < 0)
		datatable.search(searchWords().map(val => $.fn.dataTable.util.escapeRegex(val)).join('|'), true, false)
	else
		datatable.search('');
}

// this is the select
function updateFilter()
{
	var column = datatable.column(this.getAttribute('column_id'));
	setColumnSearch(this, column);

	new Promise(done =>
	{
		column.draw();
		done();
	}).then(updateFragment);
}

function updateSearch()
{
	var col = parseInt(form.querySelector('select[name="scope"]').value);

	setGlobalSearch(col);
	if (col >= 0)
		setColumnSearch(form.querySelector('select[column_id="' + col + '"]'), datatable.column(col));

	new Promise(done =>
	{
		datatable.draw();
		done();
	}).then(updateFragment);
}

// this is select.name="scope"
function updateSearchScope()
{
	var col = parseInt(this.value);

	setGlobalSearch(col);
	Array.from(this.options).map(opt => parseInt(opt.value)).forEach(val =>
	{
		if (val >= 0)
			setColumnSearch(form.querySelector('select[column_id="' + val + '"]'), datatable.column(col));
	});

	new Promise(done =>
	{
		datatable.draw();
		done();
	}).then(updateFragment);
}

async function filterUpdated(search)
{
	while (timeline.hasChildNodes())
		timeline.firstChild.remove();

	var loading = document.getElementById('loading');
	loading.style.display = 'block';

	var filteredData = datatable.rows({ filter: 'applied' }).data();
	$.each(filteredData, addToTimeline);

	loading.style.display = 'none';
}

function makeFilter(colIdx, name, sortfunction)
{
	var column = datatable.column(colIdx);
	var values = column.data().unique().sort(sortfunction);

	var opt = form.querySelector('select[name="scope"]').appendChild(document.createElement('option'));
	opt.value = colIdx;

	var p = document.createElement('p');
	p.className += 'filter_' + name
	opt.textContent = p.textContent = column.header().textContent;

	var select = p.appendChild(document.createElement('select'));
	select.multiple = true;
	select.name = name;
	select.size = values.length;
	select.setAttribute('column_id', colIdx);

	var clear = p.appendChild(document.createElement('button'));
	clear.textContent = 'clear';

	values.each(function (t)
	{
		var option = select.appendChild(document.createElement('option'));
		option.textContent = t;
		option.value = t;
	});

	select.onchange = updateFilter
	clear.onclick = () =>
	{
		select.selectedIndex = -1;
		column.search('');
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

function renderAcronym(data, type, row)
{
	if (type === 'display')
	{
		var a = data,
			i = '';
		for (y = n_years - 1; y >= 0; y--)
			if (row[cfpIdx + y * yearOffset] && row[cfpIdx + y * yearOffset] != '(missing)')
			{
				i = $('<img />').attr('src', 'wikicfplogo.png').attr('alt', 'Wiki CFP logo').addClass('cfpurl')
				.wrap('<a></a>').parent().attr('href', row[cfpIdx + y * yearOffset]).attr('title', data + ' CFP on WikiCFP');
				break;
			}

		for (y = n_years - 1; y >= 0; y--)
			if (row[linkIdx + y * yearOffset] && row[linkIdx + y * yearOffset] != '(missing)')
			{
				a = $('<a></a>').attr('href', row[linkIdx + y * yearOffset]).append(data);
				break;
			}

		return $('<p></p>').append(a).append('&nbsp;').append(i).html();
	}
	else
		return data
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


	datatable = $('#confs').DataTable(
	{
		data: json['data'],
		columns: json['columns'],
		columnDefs: [
			// acronyms: displayed wrapped in links if present, searchable
			{
				"targets": [confIdx],
				"render": renderAcronym,
				"searchable": true
			},
			// full-text titleo, rank, and field: also searchable, no special display
			{
				"targets": [titleIdx, fieldIdx, rankIdx],
				"searchable": true
			},
			// dates: display whether extrapolated, not searchable
			{
				"targets": datesIdx,
				"createdCell": markExtrapolated,
				"searchable": false
			},
			// links, booleans on extrapolated dates: hidden, not searchable
			{
				"targets": origIdx.concat(urlIdx),
				"visible": false,
				"searchable": false
			}
		],
		pageLength: 50,
		scrollY: "calc(100% - 10px)",
		order: [
			[  subIdx + (n_years - 1) * yearOffset, "asc"],
			[ abstIdx + (n_years - 1) * yearOffset, "asc"],
			[startIdx + (n_years - 1) * yearOffset, "asc"],
			[  endIdx + (n_years - 1) * yearOffset, "asc"]
		]
	});

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
	datatable.draw().on('search.dt', filterUpdated);

	var maxdate = timeline_zero;
	for (var col of datesIdx)
	{
		var colmax = datatable.column(col).data().filter(notNull).sort().reverse()[0];
		// lexical sort works thanks to format, just parse once
		maxdate = Math.max(maxdate, parse_date(colmax));
	}

	// get last day of month
	var endDate = new Date(maxdate);
	timeline_max = Date.UTC(endDate.getFullYear(), endDate.getMonth() + 1, 0);
	timeline_scale = 100 / (timeline_max - timeline_zero);

	makeTimelineLegend();
	// add data to Timeline, but only filtered
	filterUpdated();

	document.getElementById('loading').style.display = 'none';
}

$(document).ready(function ()
{
	makeTimelineLegend()

	/* $.getJSON + override mime type */
	$.ajax(
	{
		url: "cfp.json",
		dataType: "json",
		beforeSend: function (xhr)
		{
			xhr.overrideMimeType('application/json; charset="UTF-8"');
		},
		success: populatePage
	});
});
