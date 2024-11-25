app.controller("LoginCtrl", ['$scope', '$rootScope', 'factLogin', 'factComando', '$window', '$interval', 'factFocus', function ($scope, $rootScope, factLogin, factComando, $window, $interval, factFocus) {
    function _initLogin() {
        /* Aqui vai tudo que deve ser inicializado */
        $scope.frmLogin = {};

        focus('frmLoginusuario');
     
    }


    $scope.poefoco = function () {

        factFocus('frmLoginusuario');
    }
    // EFETUA LOGIN
    $scope.login = function () {
        if ($scope.frmLogin.usuario != null && $scope.frmLogin.senha != null) {

            $("#modalLoading").modal("show");

            factLogin.efetuarLogin($scope.frmLogin.usuario, $scope.frmLogin.senha).then(function (d) {
                if (d.data.d.status === true) {
                    $rootScope.idLogin = d.data.d.IdLogin;
                    $rootScope.userLogin = d.data.d.BsccLogin;
                    $rootScope.userPrivilegio = d.data.d.CsccPrivilegio;
                    $rootScope.userConveniado = d.data.d.CsccConveniado;
                    $rootScope.usuario = d.data.d.CUsuario;
                }
                else {
                    $rootScope.modalmessage = d.data.d.msg;
                    $("#modalDinamico").modal("show");
                }
            $("#modalLoading").modal("hide");
            });
        }
        else {
            $rootScope.modalmessage = "Todos os campos devem ser preenchidos";
            $("#modalDinamico").modal("show");
        }
        
    }

    // EFETUA LOGOFF 
    $scope.logout = function () {
        factLogin.efetuarLogoff().then(function (d) {
            if (d.data.d.status === true) {
                $rootScope.pagina = 'Index';
                $rootScope.userLogin = 'false';
                $rootScope.userPrivilegio = 'guest';
                $rootScope.userConveniado = [];
             //   $rootScope.usuario = '';
            }
            else {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalDinamico").modal("show");
            }
        });
    }

    _initLogin();
    $scope.initLogin = _initLogin;
}]);