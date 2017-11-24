var datatable, today = new Date();
var timeline_zero = Date.UTC(today.getFullYear(), today.getMonth() - 6, 1),
	timeline_max = Date.UTC(today.getFullYear(), today.getMonth() + 18, 1);
// px per month
var timeline_scale = 50 / Date.UTC(1970, 1, 1)
// the value we push into the hash
var sethash = '';

var ranks = ['D', 'C', 'B', 'A', 'A*'];
var confIdx = 0, titleIdx = 1, rankIdx = 2, fieldIdx = 3, origOffset = 6, linkIdx = 16, cfpIdx = 17;
var abstIdx = 4, subIdx = 5, notifIdx = 6, camIdx = 7, startIdx = 8, endIdx = 9;
var datesIdx = [4, 5, 6, 7, 8, 9], origIdx = [10, 11, 12, 13, 14, 15];

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
	var params = $("#filters select").serializeArray().map(function (item)
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

	var pm = $('<p id="months"></p>').prependTo(box),
		last = timeline_zero,
		month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
		m;

	$('<span class="acronyms">&nbsp;</span>').appendTo(pm);
	for (m = today.getMonth() - 6; last < endDate; m++)
	{
		var next = Date.UTC(today.getFullYear(), m + 1, 1),
			month = new Date(last).getMonth();

		$('<span></span>').append(month_name[month])
			.addClass((month == 11 ? 'december' : ''))
			.css('width', (next - last) * timeline_scale - (month == 11 ? 2 : 1)).appendTo(pm);

		last = next;
	}

	var py = $('<p id="years"></p>').prependTo(box);
	$('<span class="acronyms">&nbsp;</span>').appendTo(py);

	for (var y = startDate.getFullYear(); y <= endDate.getFullYear(); y++)
	{
		var from = Math.max(timeline_zero, Date.UTC(y, 0, 1));
		var to = Math.min(last, Date.UTC(y + 1, 0, 1));
		$('<span></span>').append(y).css('width', (to - from) * timeline_scale - 2).appendTo(py);
	}

	$('#timeline_body').on('scroll', function (obj)
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

function tooltipPosition(date, obj)
{
	var t = 200 * (date - timeline_zero) / (timeline_max - timeline_zero) - 100;
	if (t < 0)
		obj.css('left', (-t) + '%');
	else
		obj.css('right', t + '%');
}

function addToTimeline(n, data)
{
	var s = timeline_scale,
		last = timeline_zero,
		tooltip, filler, block;
	var acronym = data[confIdx];
	var p = $('<p></p>').append($('<span class="acronyms"></span>').append(renderAcronym(acronym, 'display', data)));
	var marker = $('<sup>†</sup>').css('width', '.5em').css('margin-right', '-.5em');
	var punctualMarker = marker.clone().css('width', '.5em').css('margin-left', '.5em').css('margin-right', '-1em');

	var abst = parse_date(data[abstIdx]);
	var submit = parse_date(data[subIdx]);
	var notif = parse_date(data[notifIdx]);

	if (!abst) abst = submit;
	else if (!submit) submit = abst;

	if (submit && notif && submit >= last && notif >= submit)
	{
		if (submit > abst)
		{
			tooltip = $('<span></span>').addClass('tooltip').append(acronym + ' registration ' + data[abstIdx]);
			filler = $('<span class="filler"></span>').css('width', (abst - last) * s);
			block = $('<span class="abstract"></span>').css('width', (submit - abst) * s);

			p.append(filler).append(block.append(tooltip));

			if (data[abstIdx + origOffset] === false)
			{
				block.append(marker.clone());
				tooltip.prepend('Estimated ');
			}

			tooltipPosition((abst + submit) / 2, tooltip);
			last = submit;
		}

		tooltip = $('<span></span>').addClass('tooltip').append(acronym + ' submission ' + data[subIdx] + ',<br />')
		filler = $('<span class="filler"></span>').css('width', (submit - last) * s);
		block = $('<span class="review"></span>').css('width', (notif - submit) * s);

		if (data[subIdx + origOffset] === false)
		{
			tooltip.prepend('Estimated ');
			block.append(marker.clone());
		}
		p.append(filler).append(block.append(tooltip));

		if (data[notifIdx + origOffset] === false)
		{
			tooltip.append('estimated ');
			p.append(marker.clone());
		}

		tooltipPosition((submit + notif) / 2, tooltip);
		tooltip.append('notification ' + data[notifIdx]);
		last = notif;

	}
	else if (submit && submit >= last)
	{
		tooltip = $('<span></span>').addClass('tooltip').append(acronym + ' submission ' + data[subIdx]);
		filler = $('<span class="filler"></span>').css('width', (submit - last) * s);
		block = $('<span class="submit"><sup>◆</sup></span>');

		p.append(filler).append(block.append(tooltip));

		if (data[subIdx + origOffset] === false)
		{
			tooltip.prepend('Estimated ');
			p.append(punctualMarker.clone());
		}

		tooltipPosition(submit, tooltip);
		last = submit;
	}

	var camera = parse_date(data[camIdx]);
	if (camera && camera >= last)
	{
		tooltip = $('<span></span>').addClass('tooltip').append(acronym + ' final version ' + data[camIdx]);
		filler = $('<span class="filler"></span>').css('width', (camera - last) * s);
		block = $('<span class="camera"><sup>∎</sup></span>');

		p.append(filler).append(block.append(tooltip));

		if (data[camIdx + origOffset] === false)
		{
			tooltip.prepend('Estimated ');
			block.append(punctualMarker.clone());
		}

		tooltipPosition(camera, tooltip);
		last = camera;
	}

	var start = parse_date(data[startIdx]);
	var end = parse_date(data[endIdx]);
	if (start && end && start >= last && end >= start)
	{
		tooltip = $('<span></span>').addClass('tooltip');
		filler = $('<span class="filler"></span>').css('width', (start - last) * s);
		block = $('<span class="conf"></span>').css('width', (end - start) * s);

		p.append(filler).append(block.append(tooltip));

		if (data[startIdx + origOffset] === false || data[endIdx + origOffset] == false)
		{
			tooltip.append(acronym + ' estimated from ' + data[startIdx] + ' to ' + data[endIdx]);
			p.append(marker.clone());
		}
		else
		{
			tooltip.append(acronym + ' from ' + data[startIdx] + ' to ' + data[endIdx]);
		}

		tooltipPosition((start + end) / 2, tooltip);
		last = end;
	}

	// ignore conferences for which we don't have a single date
	if (last > timeline_zero)
	{
		$('<span class="filler"></span>').css('width', (timeline_max - last) * s).appendTo(p);
		$('#timeline').append(p);
	}
}

function filterUpdated(search)
{
	var filteredData = datatable.rows({ filter: 'applied' }).data();
	$('#timeline').empty()
	$.each(filteredData, addToTimeline)
}

function makeFilter(column, name, initFilters, sortfunction)
{
	var values = column.data().unique().sort(sortfunction);

	var select = $('<select multiple></select>').attr("name", name).attr('size', Math.min(10, values.length));
	$("#filters").append($('<p></p>').append(column.header().innerHTML).append(select).append(
		$('<button>clear</button>').click(function ()
		{
			select.val([]).change()
		})
	));

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
		column.search(regex, true, false).draw();
		updateFragment();
	});

	if (selected.length)
		select.change();
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
		if (row[cfpIdx] && row[cfpIdx] != '(missing)')
			i = $('<img />').attr('src', 'wikicfplogo.png').attr('alt', 'Wiki CFP logo').addClass('cfpurl')
			.wrap('<a></a>').parent().attr('href', row[cfpIdx]).attr('title', data + ' CFP on WikiCFP');
		if (row[linkIdx] && row[linkIdx] != '(missing)')
			a = $('<a></a>').attr('href', row[linkIdx]).append(data);

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
	datatable = $('#confs').DataTable(
	{
		data: json['data'],
		columns: json['columns'],
		columnDefs: [
			// links, searchable but not displayed
			{
				"targets": [linkIdx, cfpIdx],
				"visible": false,
				"searchable": true
			},
			// acronyms, displayed wrapped in links if present
			{
				"targets": [confIdx],
				"render": renderAcronym
			},
			// dates, display whether extrapolated
			{
				"targets": datesIdx,
				"createdCell": markExtrapolated
			},
			// booleans on extrapolated dates, hiddent
			{
				"targets": origIdx,
				"visible": false,
				"searchable": false
			}
		],
		pageLength: 50,
		order: [
			[  subIdx, "asc"],
			[ abstIdx, "asc"],
			[startIdx, "asc"],
			[  endIdx, "asc"]
		]
	});

	$('#head').append(' The last scraping took place on ' + json['date'] + '.')

	show = $('<button>&gt;</button>').wrap('<label id="show_filters">Filters</label>');
	filter = $('<div id="filters"></div>');
	$('#confs_filter').append(filter).prepend(show.parent());

	show.click(function ()
	{
		filter.slideToggle();
		show.toggleClass('rotated');
	});

	var initFilters = parseFragment();

	makeFilter(datatable.column(confIdx), "conf", initFilters);
	makeFilter(datatable.column(rankIdx), "core", initFilters, ranksort);
	makeFilter(datatable.column(fieldIdx), "field", initFilters);

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
	$('#timeline').width('calc(8.5em + ' + (timeline_max - timeline_zero) * timeline_scale + 'px)');

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
