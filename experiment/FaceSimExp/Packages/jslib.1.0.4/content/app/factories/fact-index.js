app.factory('factIndex', ['$http', '$rootScope', function ($http, $rootScope) {
    function initIndex() {
        obter_versaoscc({}).then(function(d) {
            $rootScope.versao = d.data.d;
        });


    }
    initIndex();

    function obter_versaoscc() {

        return $http.post('ServerSide/wsscc.asmx/obter_versaoSCCCloud', {});
    }


    return {}
}]);