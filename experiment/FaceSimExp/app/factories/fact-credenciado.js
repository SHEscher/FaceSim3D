app.factory('factCredenciado', ['$http', '$rootScope', function ($http, $rootScope) {
    function initCredenciado() {

        $("#modalLoading").modal("show");

        _obterCredenciado({}).then(function (d) {
            $rootScope.lstCredenciado = d.data.d;
            for (var i in $rootScope.lstCredenciado) {
                if ($rootScope.lstCredenciado[i].Id == -1)
                    $rootScope.lstCredenciado = {};
                else
                    !$rootScope.lstCredenciado[i].Ctag && $rootScope.lstCredenciado[i].Iissuer && $rootScope.lstCredenciado[i].Ltag ? $rootScope.lstCredenciado[i].Ctag = $rootScope.lstCredenciado[i].Iissuer.toString().preencherEsq(0, 5) + $rootScope.lstCredenciado[i].Ltag.toString().preencherEsq(0, 10) : $rootScope.lstCredenciado[i].Ctag = '';
            }

            $("#modalLoading").modal("hide");

        });
    }

    function _obterCredenciado(objDados) {
        return $http.post('ServerSide/wsscc.asmx/credenciado_obterLista', { credenciado: objDados, stringContainsOperator: true });
    };
    function _definirCredenciado(objDados) {
        return $http.post('ServerSide/wsscc.asmx/credenciado_definir', { credenciado: objDados });
    };
    function _deletarCredenciado(id) {
        return $http.post('ServerSide/wsscc.asmx/credenciado_apagar', { id: id });
    };
    function _initializeModalCredenciado(objDados) {
        var modalscope = $(modalCredenciado).scope();
        modalscope.frmModalCredenciado = {};
        modalscope.frmModalCredenciado = JSON.parse(JSON.stringify(objDados));
    };
    function _clearModalCredenciado() {
        var modalscope = $(modalCredenciado).scope();
        modalscope.frmModalCredenciado = {};
        modalscope.frmModalCredenciado.Id = "";
    };
    function _initializeConfirmCredenciado(objDados) {
        var modalscope = $(modalConfirmCredenciado).scope();
        modalscope.modalConfirmCredenciado = {};
        modalscope.modalConfirmCredenciado = JSON.parse(JSON.stringify(objDados));
    };
    function _clearConfirmCredenciado() {
        var modalscope = $(modalConfirmCredenciado).scope();
        modalscope.modalConfirmCredenciado = {};
    };
    function _validarCredenciado(objDados) {
        return $http.post('ServerSide/wsscc.asmx/fachada_obterTag', { Cplaca: objDados.Cplaca });
    };
    function _validarPlacaTAG(Cplaca) {
        return $http.post('ServerSide/wsscc.asmx/cadtag_obterTag', { Cplaca: Cplaca });
    };

    initCredenciado();
    return {
        //Métodos expostos na factory, métodos que não estão listados aqui não serão acessíveis de fora.
        obterCredenciado: _obterCredenciado,
        definirCredenciado: _definirCredenciado,
        deletarCredenciado: _deletarCredenciado,
        initializeModalCredenciado: _initializeModalCredenciado,
        clearModalCredenciado: _clearModalCredenciado,
        initializeConfirmCredenciado: _initializeConfirmCredenciado,
        clearConfirmCredenciado: _clearConfirmCredenciado,
        validarCredenciado: _validarCredenciado,
        validarPlacaTAG: _validarPlacaTAG
    }
}]);