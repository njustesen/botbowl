var gulp = require('gulp');
var concat = require('gulp-concat');
var less = require('gulp-less');
var watch = require('gulp-watch');

var paths = {
    scripts: ['static/js/*.js'],
    styles: ['static/less/style.less']
};

gulp.task('scripts', function() {
    // concat and copy all JavaScript
    return gulp.src(paths.scripts)
        //.pipe(watch(paths.scripts))
        .pipe(concat('botbowl.js'))
        .pipe(gulp.dest('static/dist/js'));
});

gulp.task('styles', function() {
	return gulp.src(paths.styles)
	    .pipe(watch(paths.styles))
		.pipe(less())
		.pipe(gulp.dest('static/dist/css'));
});

// The default task (called when you run `gulp` from cli)
gulp.task('default', ['scripts', 'styles']);
