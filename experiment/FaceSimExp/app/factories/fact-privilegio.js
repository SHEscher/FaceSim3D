app.factory('factPrivilegio', ['$http', '$rootScope', function ($http, $rootScope) {
    function initPrivilegio() {

        $("#modalLoading").modal("show");

        _obterPrivilegio({ Cnome: "" }).then(function (d) {
            $rootScope.lstPrivilegio = d.data.d;

            $("#modalLoading").modal("hide");

        });
    }

    function _obterPrivilegio(objDados) {
        return $http.post('ServerSide/wsscc.asmx/privilegio_obterLista', { privilegio: objDados, stringContainsOperator: true });
    };
    function _definirPrivilegio(objDados) {
        return $http.post('ServerSide/wsscc.asmx/privilegio_definir', { privilegio: objDados });
    };
    function _deletarPrivilegio(id) {
        return $http.post('ServerSide/wsscc.asmx/privilegio_apagar', { id: id });
    };
    function _clearModalPrivilegio() {
        var modalscope = $(modalPrivilegio).scope();
        modalscope.frmModalPrivilegio = {};
        modalscope.frmModalPrivilegio.Id = "";
    };

    initPrivilegio();
    return {
        //Métodos expostos na factory, métodos que não estão listados aqui não serão acessíveis de fora.
        obterPrivilegio: _obterPrivilegio,
        definirPrivilegio: _definirPrivilegio,
        deletarPrivilegio: _deletarPrivilegio,
        clearModalPrivilegio: _clearModalPrivilegio
    }
}]);