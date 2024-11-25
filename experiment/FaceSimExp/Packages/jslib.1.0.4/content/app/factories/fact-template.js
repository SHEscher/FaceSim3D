app.factory('factTemplate', ['$http', '$rootScope', function ($http, $rootScope) {

    function init() {
        // Carregar as coisinhas do inicio
        
    }

    function _templateMethod1() {
        return $http.post('ServerSide/wsscc.asmx/_test1', { });  // Esse meTODO: no webservice n existe de vdd ;-)
    };

    function _templateMethod2(nome) {
        return $http.post('ServerSide/wsscc.asmx/_test2', { _nome: nome });  // Esse meTODO: no webservice n existe de vdd ;-)
    };

    init();

    return {

        //Métodos expostos na factory, métodos que não estão listados aqui não serão acessíveis de fora.
        templateMethod1: _templateMethod1,
        templateMethod2:_templateMethod2
    }
}]);