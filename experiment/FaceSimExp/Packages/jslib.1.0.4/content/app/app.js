var app = angular.module('sccApp', ['ui.bootstrap', 'angular-loading-bar', 'trNgGrid', 'ui.utils.masks', 'ui.mask','ngMessages']);

app.controller("MainCtrl", ['$scope', '$rootScope', function ($scope, $rootScope) {
     // Definição do controle INICIAL
    $rootScope.modalmessage = '';
    $rootScope.pagina = 'Index';
    $rootScope.porPagina = 25;

    $scope.setNotification = function (e) {
        //e.preventDefault();
        //$("#wrapper").toggleClass("toggled");
    };
    $scope.setPage = function (page) {
        $rootScope.pagina = page;
    };
    $scope.setRegistrosPorPagina = function (quantidade) {
        $rootScope.porPagina = quantidade;
    };
}]);