app.factory('factLogin', ['$http', '$rootScope', '$window', function ($http, $rootScope, $window) {
    function initLogin() {

        $("#modalLoading").modal("show");

        _obterSession().then(function (d) {
            if (d.data.d.status === true) {
                $rootScope.idLogin = d.data.d.IdLogin;
                $rootScope.userLogin = d.data.d.BsccLogin;
                $rootScope.userPrivilegio = d.data.d.CsccPrivilegio;
                $rootScope.userConveniado = d.data.d.CsccConveniado;
                $rootScope.usuario = d.data.d.CUsuario;
            }
            else {
                $rootScope.userLogin = 'false';
                $rootScope.userPrivilegio = 'guest';
                $rootScope.userConveniado = [];
            }

            $("#modalLoading").modal("hide");

        });
    }


    function _contemPrivilegio(privilegio) {
        var privilegioencontrado = false;
        var listaPrivilegio = $rootScope.userPrivilegio;
        if (!listaPrivilegio) { return }
        listaPrivilegio = listaPrivilegio.split(",");
        for (var i = 0 ; i <= listaPrivilegio.length ; i++)
        {
            if (privilegio == listaPrivilegio[i])
                privilegioencontrado = true;
        }
        return (privilegioencontrado);
    } $rootScope.contemPrivilegio = _contemPrivilegio;
    function _definirSession(usuario) {
        return $http.post('ServerSide/wsscc.asmx/definir_session', {_Usuario: usuario});
    };
    function _obterSession() {
        return $http.post('ServerSide/wsscc.asmx/obter_session', {});
    };
    function _efetuarLogin(usuario, senha) {
        return $http.post('ServerSide/wsscc.asmx/efetuar_login', { __usuario: usuario, __senha: senha });
    };
    function _efetuarLogoff() {
        return $http.post('ServerSide/wsscc.asmx/efetuar_logoff', {});
    };

    initLogin();
    return {
        //Métodos expostos na factory, métodos que não estão listados aqui não serão acessíveis de fora.
        definirSession: _definirSession,
        obterSession: _obterSession,
        efetuarLogin: _efetuarLogin,
        efetuarLogoff: _efetuarLogoff
    }
}]);