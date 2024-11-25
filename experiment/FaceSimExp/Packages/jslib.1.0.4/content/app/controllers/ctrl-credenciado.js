app.controller("CredenciadoCtrl", ['$scope', '$rootScope', 'factCredenciado', function ($scope, $rootScope, factCredenciado) {
    function _initCredenciado() {
        /* Aqui vai tudo que deve ser inicializado */
        $scope.frmFiltroValues = {};
        /* Variáveis de controle */
        $scope.accordion = {
            isFirstOpen: true,
            isFilterOpen: false
        };
    }

    //BOTAO VISUALIZAR
    $scope.visualizar = function (objDados) {
        !objDados.Ctag ? objDados.Ctag = objDados.Iissuer.toString().preencherEsq(0, 5) + objDados.Ltag.toString().preencherEsq(0, 10) : objDados.Ctag;
        $scope.frmModalCredenciado = JSON.parse(JSON.stringify(objDados));
        $scope.frmModalCredenciado.Cativo = 'Ativo';
        $("#modalCredenciado").modal("show");
    };

    //BOTAO VISUALIZAR CONFIRMAR DELETE
    $scope.visualizarConfirmarDelete = function (id) {
        $scope.idDelete = id;
        $scope.modalmessage = "Confirma exclusão do registro?";
        $("#modalConfirm").modal("show");
    };

    //BOTAO DELETAR
    $scope.deletar = function (id) {
        factCredenciado.deletarCredenciado(id).then(function (d) {
            if (d.data.d.status == false) {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalConfirm").modal("hide");
                $("#modalDinamico").modal("show");
            }
            else {
                $rootScope.modalmessage = d.data.d.msg;
                var query = Enumerable.From($rootScope.lstCredenciado);
                query = query.Where(function (x) { return x.Id === id }).Select(function (x) { return x; }).First();
                var idxobj = $rootScope.lstCredenciado.indexOf(query);
                $rootScope.lstCredenciado.splice(idxobj, 1);
                $("#modalConfirm").modal("hide");
                $("#modalDinamico").modal("show");

            }
        });
    };

    //BOTAO SUBMIT FILTRO
    $scope.submitFormFiltro = function () {
        if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = '';
        if (!$scope.frmFiltroValues.Cplaca) $scope.frmFiltroValues.Cplaca = '';
        if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = '';
        //$scope.gridOptions = {
        //    filterOptions: { filterText: '' }
        //};
        factCredenciado.obterCredenciado($scope.frmFiltroValues).then(function (d) {
            $rootScope.lstCredenciado = d.data.d;
            for (var i in $rootScope.lstCredenciado) { !$rootScope.lstCredenciado[i].Ctag && $rootScope.lstCredenciado[i].Iissuer && $rootScope.lstCredenciado[i].Ltag ? $rootScope.lstCredenciado[i].Ctag = $rootScope.lstCredenciado[i].Iissuer.toString().preencherEsq(0, 5) + $rootScope.lstCredenciado[i].Ltag.toString().preencherEsq(0, 10) : $rootScope.lstCredenciado[i].Ctag = ''; }
            $scope.accordion.isFirstOpen = true;
        }, function (d) {
            console.log(d.data.d)
        }
        );
    };

    //BOTAO LIMPAR FILTRO
    $scope.limpaFormFiltro = function () {
        $scope.frmFiltroValues = {};
    };

    //BOTAO NOVO CREDENCIADO
    $scope.visualizarNovoCredenciado = function () {
        factCredenciado.clearConfirmCredenciado();

        $("#modalConfirmCredenciado").modal("show");
    };

    //ITEMss NOVA EMPRESA
    $scope.visualizarNovoEmpresa = function () {
        factEmpresa.clearModalEmpresa();
        $("#modalEmpresa").modal("show");
    };

    //VALIDA PLACA (modalConfirmCredenciado)
    $scope.validarPlacaTAGconfirm = function (_Cplaca) {
        factCredenciado.validarPlacaTAG(_Cplaca).then(function (d) {
            if (d.data.d.status === false) {
                var _Ctag = d.data.d.msg;
                $scope.modalConfirmCredenciado.Ctag = _Ctag;
            }
            else {
                //Emissor
                var IIssuer = d.data.d.IIssuer;
                IIssuer = IIssuer.toString().preencherEsq(0, 5);
                //Tag
                var Ltag = d.data.d.LTag;
                Ltag = Ltag.toString().preencherEsq(0, 10);
                //Tag Formatada
                var _Ctag = IIssuer + Ltag
                $scope.modalConfirmCredenciado.Ctag = _Ctag;
            }
        });
    };

    //VALIDA PLACA (frmModalCredenciado)
    $scope.validarPlacaTAGmodal = function (_Cplaca) {
        factCredenciado.validarPlacaTAG(_Cplaca).then(function (d) {
            if (d.data.d.status === false) {
                var _Ctag = d.data.d.msg;
                $scope.frmModalCredenciado.Ctag = _Ctag;
            }
            else {
                //Emissor
                var IIssuer = d.data.d.IIssuer;
                IIssuer = IIssuer.toString().preencherEsq(0, 5);
                //Tag
                var Ltag = d.data.d.LTag;
                Ltag = Ltag.toString().preencherEsq(0, 10);
                //Tag Formatada
                var _Ctag = IIssuer + Ltag
                $scope.frmModalCredenciado.Ctag = _Ctag;
            }
        });
    };

    //BOTAO SUBMIT (modalConfirmCredenciado)
    $scope.submitConfirmCredenciado = function () {

        $("#modalLoading").modal("show");
        factCredenciado.validarCredenciado($scope.modalConfirmCredenciado).then(function (d) {
            if (d.data.d.status == false) {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalDinamico").modal("show");
            }
            else {
                if (d.data.d.numeroDocumento != $scope.modalConfirmCredenciado.Ccnpj_cpf) {
                    $scope.modalConfirmCredenciado.Ccnpj_cpf.length > 11 ? $rootScope.modalmessage = "CNPJ " : $rootScope.modalmessage = "CPF ";
                    $rootScope.modalmessage += "informado não correspondente para a Placa " + $scope.modalConfirmCredenciado.Cplaca.toUpperCase();
                    $("#modalDinamico").modal("show");
                }
                else {
                    $("#modalConfirmCredenciado").modal("hide");
                    $scope.modalConfirmCredenciado.Cfonecel = d.data.d.numeroCelular;
                    $scope.modalConfirmCredenciado.Ctag = '00290' + d.data.d.numeroTag;

                    factCredenciado.clearModalCredenciado();
                    $scope.visualizar($scope.modalConfirmCredenciado)
                }
            }

            $("#modalLoading").modal("hide");
        }, function (d) {
            console.log(d.data.d);
            $("#modalLoading").modal("hide");
        });
    };

    //BOTAO SUBMIT (frmModalCredenciado)
    $scope.submitModalCredenciado = function () {
        if (!$scope.frmModalCredenciado.Id) $scope.frmModalCredenciado.Id = 0;
        if (!$scope.frmModalCredenciado.IIssuer) $scope.frmModalCredenciado.IIssuer = $scope.frmModalCredenciado.Ctag.substr(0, 5);
        if (!$scope.frmModalCredenciado.LTag) $scope.frmModalCredenciado.LTag = $scope.frmModalCredenciado.Ctag.substr(6, 10);
        //if (!$scope.frmModalCredenciado.Id_empresa) $scope.frmModalCredenciado.Id_empresa = $scope.frmModalCredenciado.Cempresa;
        if (!$scope.frmModalCredenciado.Csenha) $scope.frmModalCredenciado.Csenha = $scope.frmModalCredenciado.Ccnpj_cpf.substr(0, 5);
        $scope.frmModalCredenciado.Ccnpj_cpf.length > 11 ? $scope.frmModalCredenciado.Ctipo_documento = "CNPJ" : $scope.frmModalCredenciado.Ctipo_documento = "CPF";

        $("#modalLoading").modal("show");

        factCredenciado.definirCredenciado($scope.frmModalCredenciado).then(function (d) {
            if (d.data.d.status == false) {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalCredenciado").modal("hide");
                $("#modalDinamico").modal("show");
            }
            else {
                $rootScope.modalmessage = d.data.d.msg;
                if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = "";
                if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
                if ($scope.frmModalCredenciado.Id != 0) {
                    //DELETA ITEM ANTERIOR
                    var query = Enumerable.From($rootScope.lstCredenciado);
                    query = query.Where(function (x) { return x.Id === $scope.frmModalCredenciado.Id }).Select(function (x) { return x; }).First();
                    var idxobj = $rootScope.lstCredenciado.indexOf(query);
                    $rootScope.lstCredenciado.splice(idxobj, 1);
                }
                $rootScope.lstCredenciado.push(d.data.d);
                $("#modalCredenciado").modal("hide");
                $("#modalDinamico").modal("show");
            }
            $("#modalLoading").modal("hide");
        }), function (d) {
            console.log(d);
            $("#modalLoading").modal("hide");

        };
    };

    _initCredenciado();
    $scope.initCredenciado = _initCredenciado; //Aqui definimos o que será exposto no escopo
}]);