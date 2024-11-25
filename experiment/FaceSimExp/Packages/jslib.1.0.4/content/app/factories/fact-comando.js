app.factory('factComando', ['$http', '$rootScope', function ($http, $rootScope) {
    function initComando() {

        $("#modalLoading").modal("show");

        _obterComando({}).then(function (d) {
            $rootScope.lstComandos = d.data.d;

            $("#modalLoading").modal("hide");

        });
    }

    function _obterComando(objDados) {
        return $http.post('ServerSide/wsscc.asmx/comando_obterLista', { comando: objDados, stringContainsOperator: false });
    };

    function _comando_definir(objDados) {
        return $http.post('ServerSide/wsscc.asmx/comando_definir', { comando: objDados});
    };
    ///
    //initComando();
    return {
        //Métodos expostos na factory, métodos que não estão listados aqui não serão acessíveis de fora.
        obterComando: _obterComando,
        comando_definir: _comando_definir
    }
}]);