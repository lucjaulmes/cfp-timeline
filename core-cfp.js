var datatable, today = new Date();
var timeline_zero = Date.UTC(today.getFullYear(), today.getMonth() - 6);
// px per month
var timeline_scale = 50 / Date.UTC(1970, 1, 1)

var ranks = ['A*', 'A', 'B', 'C', 'D'];
function ranksort(a, b) { return ranks.indexOf(a) - ranks.indexOf(b); }

/* https://stackoverflow.com/a/19015262 */
function getScrollBarWidth() {
    var outer = $('<div>').css({visibility: 'hidden', width: 100, overflow: 'scroll'}).appendTo('body'),
        widthWithScroll = $('<div>').css({width: '100%'}).appendTo(outer).outerWidth();
    outer.remove();
    return 100 - widthWithScroll;
};

var offset = getScrollBarWidth();

function resizeHeaderBackground() {
	var timeline = $('#timeline'), tl_dom = timeline.get(0);
	var bar_v = tl_dom.scrollWidth > timeline.innerWidth() | 0;
	var bar_h = tl_dom.scrollHeight > timeline.innerHeight() | 0;

	$('#header').width(timeline.outerWidth(true) - offset * bar_h).height(timeline.outerHeight(true) - offset * bar_v);
}

function makeTimelineLegend(maxdate) {
	var box = $('#header')
	var startDate = new Date(timeline_zero);
	var endDate = typeof maxdate !== 'undefined' ? new Date(maxdate) :
		new Date(Date.UTC(today.getFullYear(), today.getMonth() + 12));

	// get last day of month
	// go to first day to avoid side effects of incrementing month
	endDate.setDate(1);

	// then increment month and go to day -1
	endDate.setMonth(endDate.getMonth() + 36);
	endDate.setDate(0);

	box.empty()

	var pm = $('<p id="months"></p>').prependTo(box), m, last = timeline_zero;
	var month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
	$('<span class="acronyms">&nbsp;</span>').appendTo(pm);
	for (m = today.getMonth() - 6; last < endDate; m++)
	{
		var next = Date.UTC(today.getFullYear(), m + 1), month = new Date(last).getMonth();
		$('<span></span>').append(month_name[month])
			.addClass((month == 11 ? 'december' : ''))
			.css('width', (next - last) * timeline_scale - (month == 11 ? 2 : 1)).appendTo(pm);
		last = next;
	}

	var py = $('<p id="years"></p>').prependTo(box);
	$('<span class="acronyms">&nbsp;</span>').appendTo(py);
	for (var y = startDate.getFullYear(); y <= endDate.getFullYear(); y++)
	{
		var from = Math.max(timeline_zero, Date.UTC(y, 0));
		var to = Math.min(last, Date.UTC(y, 12));
		$('<span></span>').append(y).css('width', (to - from) * timeline_scale - (to < last ? 2 : 1)).appendTo(py);
	}

	resizeHeaderBackground();
	setInterval(resizeHeaderBackground, 100);

	$('#timeline').on('scroll', function() {
		$('#months').scrollLeft($(this).scrollLeft());
		$('#years').scrollLeft($(this).scrollLeft());
	});
}

function parse_date(str) {
	if (!str) return null;
	tok = str.split('-');
	return Math.max(timeline_zero, Date.UTC(tok[0], tok[1], tok[2]));
}

function addToTimeline(n, data) {
	var s = timeline_scale, last = timeline_zero, tooltip, block;
	var p = $('<p></p>').append($('<span class="acronyms"></span>').append(data[0]));

	var abst = parse_date(data[3]);
	var submit = parse_date(data[4]);
	var notif = parse_date(data[5]);

	if (!abst) abst = submit;
	else if (!submit) submit = abst;

	if (submit && notif) {
		if (submit > abst) {
			tooltip = $('<span></span>').addClass('tooltip').append(data[0]+' registration '+data[3]);
			block = $('<span class="abstract"></span>')
				.css('margin-left', (abst - last) * s)
				.css('padding-right', (submit - abst) * s)
				.css('width', 0);

			if (data[11] === false) {
				block.html('<sup>†</sup>');
				tooltip.prepend('Estimated ');
			}

			block.append(tooltip).appendTo(p);
			last = submit;
		}

		tooltip = $('<span></span>').addClass('tooltip').append(data[0]+' submission '+data[4]+',<br />')
		block = $('<span class="review"></span>')
			.css('margin-left', (submit - last) * s)
			.css('padding-right', (notif - submit) * s)
			.css('width', 0);

		if (data[12] === false) {
			tooltip.prepend('Estimated ');
			block.append('<sup>†</sup>');
		}
		block.append(tooltip).appendTo(p);

		if (data[13] === false) {
			tooltip.append('estimated ');
			$('<span><sup>†</sup></span>').css('width', 0).appendTo(p);
		}

		tooltip.append('notification '+data[5]);
		last = notif;

	} else if (submit) {
		tooltip = $('<span></span>').addClass('tooltip').append(data[0]+' submission '+data[4]);
		block = $('<span class="submit"><sup>◆</sup></span>')
			.css('margin-left', (submit - last) * s).css('width', 0);

		if (data[12] === false) {
			tooltip.prepend('Estimated ');
			block.append('<sup>†</sup>');
		}

		block.append(tooltip).appendTo(p);
		last = submit;
	}

	var camera = parse_date(data[6]);
	if (camera) {
		tooltip = $('<span></span>').addClass('tooltip').append(data[0]+' final version '+data[6]);
		block = $('<span class="camera"><sup>∎</sup></span>')
			.css('margin-left', (camera - last) * s).css('width', 0);

		if (data[14] === false) {
			tooltip.prepend('Estimated ');
			block.append('<sup>†</sup>');
		}

		block.append(tooltip).appendTo(p);
		last = camera;
	}

	var start = parse_date(data[7]);
	var end = parse_date(data[8]);
	if (start && end) {
		tooltip = $('<span></span>').addClass('tooltip');
		block = $('<span class="conf"></span>')
			.css('margin-left', (start - last) * s)
			.css('padding-left', (end - start) * s)
			.css('width', 0);

		if (data[15] === false || data[16] == false) {
			tooltip.append(data[0]+' estimated from '+data[7]+' to '+data[8]);
			block.append('<sup>†</sup>');
		} else {
			tooltip.append(data[0]+' from '+data[7]+' to '+data[8]);
		}

		block.append(tooltip).appendTo(p);
		last = end;
	}

	// ignore conferences for which we don't have a single date
	if (last > timeline_zero) {
		$('#timeline').append(p)
	}
}

function filterUpdated(search) {
	var filteredData = datatable.rows({filter:'applied'}).data();
	$('#timeline').empty()
	$.each(filteredData, addToTimeline)
}

function makeFilter(column, sortfunction) {
	var values = column.data().unique().sort(sortfunction);

	var select = $('<select multiple></select>')
		.attr('size', Math.min(10, values.length))
		.on('change', function () {
			var val = $.map($(this).val(), $.fn.dataTable.util.escapeRegex), regex = '';
			if (val.length)
				regex = '^('+val.join('|')+')$';
			column.search(regex, true, false).draw();
		});

	values.each(function (t) {
		select.append('<option value="'+t+'">'+t+'</option>');
	});

	$('<p></p>').appendTo( $("#filters") ).append(column.header().innerHTML).append(select).append(
		$('<button>clear</button>').click(function(){ select.val([]).change() })
	);
}

function renderAcronym(data, type, row) {
	if (type === 'display' && row[10] && row[10] != '(missing)')
		return $('<a></a>').attr('href', row[10]).append(data)[0].outerHTML;
	else
		return data;
}

function markExtrapolated(td, data, rowdata, row, col) {
	if (data && rowdata[col+8] === false)
		$(td).addClass('extrapolated');
}

function notNull(val, idx) {
	return val != null
}

function populatePage(json) {
	datatable = $('#confs').DataTable({
		data: json['data'],
		columns: json['columns'],
		columnDefs: [
			// links, searchable but not displayed
			{"targets": [10], "visible": false, "searchable": true},
			// acronyms, displayed wrapped in links if present
			{"targets": [0], "render": renderAcronym},
			// dates, display whether extrapolated
			{"targets": [3, 4, 5, 6, 7, 8], "createdCell": markExtrapolated},
			// booleans on extrapolated dates, hiddent
			{"targets": [11, 12, 13, 14, 15, 16], "visible": false, "searchable": false}
		],
		pageLength: 50,
		order: [[4, "asc"], [3, "asc"], [7, "asc"], [8, "asc"]]
	});

	show = $('<button>&gt;</button>')
	filter = $('<div id="filters"></div>')

	show.click(function() {
		filter.slideToggle();
		show.toggleClass('rotated');
	});
	show.wrap('<label id="show_filters">Filters</label>');
	$('#confs_filter').append(filter).prepend(show.parent())
	makeFilter(datatable.column(0));
	makeFilter(datatable.column(2), ranksort);
	makeFilter(datatable.column(9));

	datatable.draw().on('search.dt', filterUpdated);

	$.each(datatable.rows().data(), addToTimeline)

	var maxdate = timeline_zero;
	for (var col = 3; col <= 8; col++)
	{
		var colmax = datatable.column(col).data().filter(notNull).sort().reverse()[0];
		// lexical sort works thanks to format, just parse once
		maxdate = Math.max(maxdate, parse_date(colmax));
	}

	makeTimelineLegend(maxdate);
}

$(document).ready(function() {
	makeTimelineLegend()

	/* $.getJSON + override mime type */
	$.ajax({
		url: "cfp.json",
		dataType: "json",
		beforeSend: function( xhr ) {
			xhr.overrideMimeType('application/json; charset="UTF-8"');
		},
		success: populatePage
	});
});

