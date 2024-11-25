app.factory('factContrato', ['$http', '$rootScope', function ($http, $rootScope) {
    function initContrato() {

        $("#modalLoading").modal("show");

        _obterContratoConveniado({ Id_conveniados: $rootScope.userConveniado.split(',') }).then(function (d) {
            var ret = d.data.d;
            for (i = 0; i < ret.length ; i++)
                ret[i].Ivalor = ret[i].Ivalor / 100;
            $rootScope.lstContrato = ret;

            $("#modalLoading").modal("hide");

        });

    
    }

    function _obterContratoConveniado(objDados) {
        return $http.post('ServerSide/wsscc.asmx/contrato_conveniado_obterLista', { contratoConveniado: objDados });
    };

    //teste
    function _obterContratoPorID(id) {
        return $http.post('ServerSide/wsscc.asmx/contrato_conveniado_obterListaPorContrato', { idcontrato: id });
    }

    function _obterContrato(objDados) {
        return $http.post('ServerSide/wsscc.asmx/contrato_obterLista', { contrato: objDados, stringContainsOperator: true });
    };
    function _definirContrato(objDados) {
        objDados.Ivalor = parseInt( objDados.Ivalor * 100);
        return $http.post('ServerSide/wsscc.asmx/contrato_definir', { contrato: objDados });
    };
    function _deletarContrato(id) {
        return $http.post('ServerSide/wsscc.asmx/contrato_apagar', { id: id });
    };
    function _clearModalContrato() {
        var modalscope = $(modalContrato).scope();
        modalscope.frmModalContrato = {};
        modalscope.frmModalContrato.Cativo = 'Ativo';
    };

    initContrato();
    return {
        //Métodos expostos na factory, métodos que não estão listados aqui não serão acessíveis de fora.
        obterContratoConveniado: _obterContratoConveniado,
        obterContrato: _obterContrato,
        definirContrato: _definirContrato,
        deletarContrato: _deletarContrato,
        clearModalContrato: _clearModalContrato,
        obterContratoPorID: _obterContratoPorID,
    }
}]);