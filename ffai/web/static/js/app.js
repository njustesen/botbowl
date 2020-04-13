'use strict';

var app = angular.module('app', ['ngRoute', 'ngSanitize', 'appControllers', 'appServices', 'appDirectives', 'appFilters']);

var appServices = angular.module('appServices', []);
var appControllers = angular.module('appControllers', []);
var appDirectives = angular.module('appDirectives', []);
var appFilters = angular.module('appFilters', []);

var options = {};
options.api = {};
//options.api.base_url = "http://127.0.0.1:5000";
options.api.base_url = window.location.protocol + "//" + window.location.host;

app.config(['$locationProvider', '$routeProvider', 
  function($location, $routeProvider) {
    $routeProvider.
        when('/', {
            templateUrl: 'static/partials/game.list.html',
            controller: 'GameListCtrl'
        }).
        when('/game/create', {
            templateUrl: 'static/partials/game.create.html',
            controller: 'GameCreateCtrl',
            access: { requiredAuthentication: true }
        }).
        when('/game/play/:id/:team_id', {
            templateUrl: 'static/partials/game.play.html',
            controller: 'GamePlayCtrl',
            access: { requiredAuthentication: true }
        }).
        when('/game/hotseat/:id', {
            templateUrl: 'static/partials/game.play.html',
            controller: 'GamePlayCtrl',
            access: { requiredAuthentication: true }
        }).
        when('/game/spectate/:id', {
            templateUrl: 'static/partials/game.play.html',
            controller: 'GamePlayCtrl',
            access: { requiredAuthentication: true }
        }).
        when('/game/replay/:id/', {
            templateUrl: 'static/partials/game.play.html',
            controller: 'GamePlayCtrl',
            access: { requiredAuthentication: true }
        }).
        otherwise({
            redirectTo: '/'
        });
}]);

