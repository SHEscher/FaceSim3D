app.factory('factEmpresa', ['$http', '$rootScope', function ($http, $rootScope) {
    function initEmpresa() {

        $("#modalLoading").modal("show");

        _obterEmpresaCredenciado({ Id_conveniados: $rootScope.userConveniado.split(',') }).then(function (d) {
            $rootScope.lstEmpresa = d.data.d;

            $("#modalLoading").modal("hide");

        });
    }

    function _obterEmpresa(objDados) {
        return $http.post('ServerSide/wsscc.asmx/empresa_obterLista', { empresa: objDados, stringContainsOperator: false });
    };
    function _obterEmpresaCredenciado(objDados) {
        return $http.post('ServerSide/wsscc.asmx/empresa_obterListaPorCredenciados', { empresa: objDados, stringContainsOperator: false });
    };
    function _definirEmpresa(objDados) {
        return $http.post('ServerSide/wsscc.asmx/empresa_definir', { empresa: objDados });
    };
    function _deletarEmpresa(id) {
        return $http.post('ServerSide/wsscc.asmx/empresa_apagar', { id: id });
    };
    function _clearModalEmpresa() {
        var modalscope = $(modalEmpresa).scope();
        modalscope.frmModalEmpresa = {};
        modalscope.frmModalEmpresa.Id = "";
    };

    initEmpresa();
    return {
        //Métodos expostos na factory, métodos que não estão listados aqui não serão acessíveis de fora.
        obterEmpresa: _obterEmpresa,
        obterEmpresaCredenciado: _obterEmpresaCredenciado,
        definirEmpresa: _definirEmpresa,
        deletarEmpresa: _deletarEmpresa,
        clearModalEmpresa: _clearModalEmpresa
    }
}]);