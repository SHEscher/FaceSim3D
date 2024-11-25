app.controller("CategoriaCtrl", ['$scope', '$rootScope', 'factCategoria', 'factConveniado', function ($scope, $rootScope, factCategoria, factConveniado) {
    function _initCategoria() {
        /* Aqui vai tudo que deve ser inicializado */
        $scope.frmFiltroValues = {};
        $scope.frmModalCategoria = {};
        $scope.sel_categoria_conveniado = [];   // array dos conveniados do usuario selecionado
        $scope._tab_active = "categoria";   // primeira TAB exibida

        /* Variáveis de controle */
        $scope.accordion = {
            isFirstOpen: true,
            isFilterOpen: false
        };
    }

    // SETA TAB SELECIONADA
    $scope.selecttab = function (settab) {
        $scope._tab_active = settab;
    };

    // BOTAO VISUALIZAR CATEGORIA
    $scope.visualizar = function (objDados) {
        $scope.frmModalCategoria = {};
        $scope._tab_active = "categoria";
        $scope.frmModalCategoria = JSON.parse(JSON.stringify(objDados));
        $scope.frmModalCategoria.Ivalor = $scope.frmModalCategoria.Ivalor / 100;

        //Remove ':' do horario para exibição correta dentro do modal de horários
        $scope.frmModalCategoria.Csegini = $scope.frmModalCategoria.Csegini.replace(':', '');
        $scope.frmModalCategoria.Csegfim = $scope.frmModalCategoria.Csegfim.replace(':', '');
        $scope.frmModalCategoria.Cterini = $scope.frmModalCategoria.Cterini.replace(':', '');
        $scope.frmModalCategoria.Cterfim = $scope.frmModalCategoria.Cterfim.replace(':', '');
        $scope.frmModalCategoria.Cquaini = $scope.frmModalCategoria.Cquaini.replace(':', '');
        $scope.frmModalCategoria.Cquafim = $scope.frmModalCategoria.Cquafim.replace(':', '');
        $scope.frmModalCategoria.Cquiini = $scope.frmModalCategoria.Cquiini.replace(':', '');
        $scope.frmModalCategoria.Cquifim = $scope.frmModalCategoria.Cquifim.replace(':', '');
        $scope.frmModalCategoria.Csexini = $scope.frmModalCategoria.Csexini.replace(':', '');
        $scope.frmModalCategoria.Csexfim = $scope.frmModalCategoria.Csexfim.replace(':', '');
        $scope.frmModalCategoria.Csabini = $scope.frmModalCategoria.Csabini.replace(':', '');
        $scope.frmModalCategoria.Csabfim = $scope.frmModalCategoria.Csabfim.replace(':', '');
        $scope.frmModalCategoria.Cdomini = $scope.frmModalCategoria.Cdomini.replace(':', '');
        $scope.frmModalCategoria.Cdomfim = $scope.frmModalCategoria.Cdomfim.replace(':', '');
        $scope.frmModalCategoria.Cferini = $scope.frmModalCategoria.Cferini.replace(':', '');
        $scope.frmModalCategoria.Cferfim = $scope.frmModalCategoria.Cferfim.replace(':', '');

        $("#modalCategoria").modal("show");
    };

    //NOVA CATEGORIA
    $scope.visualizarNovaCategoria = function () {
        factCategoria.clearModalCategoria();

       // $scope.filtraconveniado();
        //Inicializa horarios default das categorias
        $scope.frmModalCategoria.Csegini = "0000";
        $scope.frmModalCategoria.Csegfim = "2359";
        $scope.frmModalCategoria.Cterini = "0000";
        $scope.frmModalCategoria.Cterfim = "2359";
        $scope.frmModalCategoria.Cquaini = "0000";
        $scope.frmModalCategoria.Cquafim = "2359";
        $scope.frmModalCategoria.Cquiini = "0000";
        $scope.frmModalCategoria.Cquifim = "2359";
        $scope.frmModalCategoria.Csexini = "0000";
        $scope.frmModalCategoria.Csexfim = "2359";
        $scope.frmModalCategoria.Csabini = "0000";
        $scope.frmModalCategoria.Csabfim = "2359";
        $scope.frmModalCategoria.Cdomini = "0000";
        $scope.frmModalCategoria.Cdomfim = "2359";
        $scope.frmModalCategoria.Cferini = "0000";
        $scope.frmModalCategoria.Cferfim = "2359";

        $("#modalCategoria").modal("show");
    };

    // BOTAO VISUALIZAR CONFIRMAR DELETE CATEGORIA    
    $scope.visualizarConfirmarDelete = function (id) {
        $scope.idDelete = id;
        $scope.modalmessage = "Confirma exclusão do registro?";
        $("#modalConfirm").modal("show");
    };

    // BOTAO DELETAR CATEGORIA
    $scope.deletar = function (id) {
        factCategoria.deletarCategoria(id).then(function (d) {
            if (d.data.d.status == false) {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalConfirm").modal("hide");
                $("#modalDinamico").modal("show");
            }
            else {
                $rootScope.modalmessage = d.data.d.msg;
                var query = Enumerable.From($rootScope.lstCategoria);
                query = query.Where(function (x) { return x.Id === id }).Select(function (x) { return x; }).First();
                var idxobj = $rootScope.lstCategoria.indexOf(query);
                $rootScope.lstCategoria.splice(idxobj, 1);
                $("#modalConfirm").modal("hide");
                $("#modalDinamico").modal("show");
            }
        });
    };

    // BOTAO SUBMIT FILTRO
    $scope.submitFormFiltro = function () {
        if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = "";
        factCategoria.obterCategoria($scope.frmFiltroValues).then(function (d) {
            $rootScope.lstCategoria = d.data.d;
            //$rootScope.lstCategoria.Ivalor = $rootScope.lstCategoria.Ivalor / 100;
            $scope.accordion.isFirstOpen = true;
        }, function (r) {
            alert(r);
        });
    };
    //
    //BOTAO LIMPAR FILTRO
    $scope.limpaFormFiltro = function () {
        $scope.frmFiltroValues = {};
    };

    // BOTAO INSERIR/ATUALIZAR CATEGORIA
    $scope.verificaacao = function () {
        if (!$scope.frmModalCategoria.Id) {
            $scope.frmModalCategoria.Id = 0;

            $scope.frmModalCategoria.Ivalor = ($scope.frmModalCategoria.Ivalor).toFixed(2) * 100;
            factCategoria.definirCategoria($scope.frmModalCategoria).then(function (d) {

                if (d.data.d.status == false) {
                    $rootScope.modalmessage = d.data.d.msg;
                    $("#modalCategoria").modal("hide");
                    $("#modalDinamico").modal("show");
                }
                else {
                    $rootScope.modalmessage = d.data.d.msg;
                    if (!$scope.frmModalCategoria.Id_conveniado) $scope.frmModalCategoria.Id_conveniado = 0;

                    factCategoria.obterCategoria({}).then(function (d) {
                        $rootScope.lstCategoria = d.data.d;
                        $scope.accordion.isFirstOpen = true;
                    });
                    $("#modalCategoria").modal("hide");
                    $("#modalDinamico").modal("show");
                }
            });
        }
        else {
            $scope.frmModalCategoria.Ivalor = $scope.frmModalCategoria.Ivalor * 100;

            factCategoria.definirCategoria($scope.frmModalCategoria).then(function (d) {

                if (d.data.d.status == false) {
                    $rootScope.modalmessage = d.data.d.msg;
                    $("#modalCategoria").modal("hide");
                    $("#modalDinamico").modal("show");
                }
                else {
                    $rootScope.modalmessage = d.data.d.msg;

                    //if (!$scope.frmModalCategoria.Cnome) $scope.frmModalCategoria.Cnome = "";                   
                    factCategoria.obterCategoria({}).then(function (d) {
                        $rootScope.lstCategoria = d.data.d;
                        $scope.accordion.isFirstOpen = true;
                    });
                    $("#modalCategoria").modal("hide");
                    $("#modalDinamico").modal("show");
                }
            });
        }
    };

  

    _initCategoria();
    $scope.initCategoria = _initCategoria; //Aqui definimos o que será exposto no escopo
}]);