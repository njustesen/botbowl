appFilters.filter('range', function() {
  return function(input, total) {
    total = parseInt(total);
    for (var i=0; i<total; i++)
      input.push(i);
    return input;
  };
});

appFilters.filter('board', function() {
  return function(input) {
    total = parseInt(total);
    for (var i=0; i<total; i++)
      input.push(i);
    return input;
  };
});

app.filter('reverse', function() {
  return function(items) {
    return items.slice().reverse();
  };
});