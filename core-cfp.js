const ranks = ['D', 'C', 'B', 'A', 'A*'];
const confIdx = 0, titleIdx = 1, rankIdx = 2, fieldIdx = 3, linkIdx = 16, cfpIdx = 17;
const abstIdx = 4, subIdx = 5, notifIdx = 6, camIdx = 7, startIdx = 8, endIdx = 9;
const yearIdx = 4, yearOffset = 14, origOffset = 6;
const today = new Date(), year = today.getFullYear();

const timeline_zero = Date.UTC(today.getFullYear(), today.getMonth() - 6, 1);

// some global variables
var timeline_max = Date.UTC(today.getFullYear(), today.getMonth() + 18, 1);
// % per month: 50px / duration of 1 month
var timeline_scale = 100 / (timeline_max - timeline_zero)

var datatable, timeline = document.getElementById('timeline'), n_years = 1;
var timeline_dom_cache = {};
var filters = document.getElementById('filters');

// the value we push into the hash
var sethash = '';


function ranksort(a, b)
{
	return ranks.indexOf(b) - ranks.indexOf(a);
}

function parseFragment()
{
	var hash = window.location.hash.substr(1).split('&');
	var goto = null;

	var result = hash.reduce(function (result, item)
	{
		var parts = item.split('=', 2);

		if (parts.length > 1)
		{
			if (!result[parts[0]])
				result[parts[0]] = [];

			result[parts[0]].push(decodeURIComponent(parts[1]));
		}
		else if (item && $('#' + item).length)
			goto = item;

		return result;
	}, {});

	if (result.length && goto)
		$('html, body').scrollTop($('#' + goto).offset().top);

	return result;
}

function updateFragment()
{
	var params = $(filters).find('select').serializeArray().map(function (item)
	{
		return item['name'] + '=' + item['value']
	});
	params = params.sort().filter(function (it, pos, arr)
	{
		return !pos || it != arr[pos - 1];
	});

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
	var box = $('#timeline_header');
	var startDate = new Date(timeline_zero),
		endDate = new Date(timeline_max);

	box.empty()

	var pm = $('<p id="months"><span class="acronyms">&nbsp;</span><span class="timeblocks"></span></p>').prependTo(box).find('.timeblocks'),
		last = timeline_zero,
		month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
		m;

	for (m = today.getMonth() - 6; last < endDate; m++)
	{
		var date = Date.UTC(today.getFullYear(), m, 1),
			next = Date.UTC(today.getFullYear(), m + 1, 1),
			month = new Date(last).getMonth();

		$('<span></span>').append(month_name[month])
			.addClass(m == today.getMonth() - 6 || month == 0 ? 'first' : '')
			.css('width', (next - date) * timeline_scale + '%')
			.css('left', (date - timeline_zero) * timeline_scale + '%').appendTo(pm);

		last = next;
	}

	var py = $('<p id="years"><span class="acronyms">&nbsp;</span><span class="timeblocks"></span></p>').prependTo(box).find('.timeblocks');

	for (var y = startDate.getFullYear(); y <= endDate.getFullYear(); y++)
	{
		var from = Math.max(timeline_zero, Date.UTC(y, 0, 1));
		var to = Math.min(last, Date.UTC(y + 1, 0, 1));
		$('<span></span>').append(y).css('width', (to - from) * timeline_scale + '%').appendTo(py)
					.css('left', (from - timeline_zero) * timeline_scale + '%').appendTo(py);
	}

	$('#timeline').on('scroll', function (obj)
	{
		$('#timeline_header').scrollLeft(obj.currentTarget.scrollLeft);
	});
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
	span.style.left = 'calc(' + (date - timeline_zero) * timeline_scale + 'px - .25em)';

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

async function filterUpdated(search)
{
	while (timeline.hasChildNodes())
		timeline.firstChild.remove();

	$('#loading').show();
	var filteredData = datatable.rows({ filter: 'applied' }).data();
	$.each(filteredData, addToTimeline);
	$('#loading').hide();
}

function makeFilter(column, name, initFilters, sortfunction)
{
	var values = column.data().unique().sort(sortfunction);

	var select = $('<select multiple></select>').attr('name', name).attr('size', values.length);
	var p = $('<p></p>').append(column.header().innerHTML).append(select).append(
		$('<button>clear</button>').click(function ()
		{
			select.val([]).change()
		})
	);

	var selected = initFilters[name] ? initFilters[name] : [];
	values.each(function (t)
	{
		select.append('<option value="' + t + '"' + (selected.indexOf(t) < 0 ? '' : ' selected="selected"') + '>' + t + '</option>');
	});

	select.on('change', function ()
	{
		var val = $.map($(this).val(), $.fn.dataTable.util.escapeRegex),
			regex = '';
		if (val.length)
			regex = '^(' + val.join('|') + ')$';
		column.search(regex, true, false);
		new Promise(done => {column.draw(); done();}).then(() => updateFragment());
	});

	if (selected.length)
		select.change();

	return p[0];
}

function updateFilters()
{
	var filters = parseFragment();
	$('#filters select').each(function (i, obj)
	{
		var sel = $(obj);
		sel.val(filters[sel.attr('name')] || []).change();
	})
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
		$(td).addClass('extrapolated');
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

	$('#head').append(' The last scraping took place on ' + json['date'] + '.')

	var initFilters = parseFragment();

	filters.appendChild(makeFilter(datatable.column(confIdx), "conf", initFilters));
	filters.appendChild(makeFilter(datatable.column(rankIdx), "core", initFilters, ranksort));
	filters.appendChild(makeFilter(datatable.column(fieldIdx), "field", initFilters));

	updateFragment();

	$(window).on('hashchange', updateFilters);
	datatable.draw().on('search.dt', filterUpdated);

	var maxdate = timeline_zero;
	for (var col of datesIdx)
	{
		var colmax = datatable.column(col).data().filter(notNull).sort().reverse()[0];
		// lexical sort works thanks to format, just parse once
		maxdate = Math.max(maxdate, parse_date(colmax));
	}
	var endDate = new Date(maxdate);

	// get last day of month
	// go to first day to avoid side effects of incrementing month
	endDate.setDate(1);
	endDate.setMonth(endDate.getMonth() + 1);
	endDate.setDate(0);
	timeline_max = endDate.getTime();
	//$('#timeline').width('calc(8.5em + ' + (timeline_max - timeline_zero) * timeline_scale + 'px)');

	makeTimelineLegend();
	// add data to Timeline, but only filtered
	filterUpdated();

	$('#loading').hide()
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
