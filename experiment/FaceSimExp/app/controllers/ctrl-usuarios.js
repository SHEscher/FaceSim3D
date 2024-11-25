app.controller("UsuariosCtrl", ['$scope', '$rootScope', 'factUsuarios', 'factPrivilegio', 'factConveniado', 'factLogin', 'factFocus', function ($scope, $rootScope, factUsuarios, factPrivilegio, factConveniado, factLogin, factFocus) {
    function _initUsuario() {
        /* Aqui vai tudo que deve ser inicializado */
        $scope.frmFiltroValues = {};        // 
        $scope.sel_usuario_privilegio = []; // array dos privilegios do usuario selecionado
        $scope.sel_usuario_conveniado = []; // array dos conveniados do usuario selecionado
        $scope._tab_active = "usuarios";    // primeira TAB exibida
        $scope.form = {};

        /* Variáveis de controle */
        $scope.accordion = {
            isFirstOpen: true,
            isFilterOpen: false
        };
        /*  variaveis de controle de validacao da tab modalusuario */
        $scope.isvalidtab1 = true;
        $scope.isvalidtab2 = true;
        $scope.isvalidtab3 = true;
      

        //Obtém lista de conveniados para evitar bug de listagem quando um filtro é realizado na pagina conveniado
        //factConveniado.obterConveniado($scope.frmFiltroValues).then(function (d) {
        //    $rootScope.lstConveniado = d.data.d;
        //    $scope.accordion.isFirstOpen = true;
        //});


    }

    // SETA TAB SELECIONADA
    $scope.selecttab = function (settab) {
        $scope._tab_active = settab;
    };

    // BOTAO VISUALIZAR USUARIO
    $scope.visualizar = function (objDados) {
        $scope.mfrmFiltroValues = {};

        /*  variaveis de controle de validacao da tab modalusuario */
        $scope.isvalidtab1 = true;
        $scope.isvalidtab2 = true;
        $scope.isvalidtab3 = true;
    
        $scope._tab_active = "usuarios";
        $scope.sel_usuario_privilegio = [];
        $scope.sel_usuario_conveniado = [];
        $scope.mfrmFiltroValues = JSON.parse(JSON.stringify(objDados));
        $scope.sel_usuario_privilegio = $scope.mfrmFiltroValues.tb_usuario_privilegio.slice(0);
        $scope.sel_usuario_conveniado = $scope.mfrmFiltroValues.tb_conveniado_usuario.slice(0);
        $scope.mfrmFiltroValues.searchText = "";
        // $scope.sel_usuario_conveniado = $rootScope.lstConveniado;
        $("#modalUsuario").modal("show");
    };

    // BOTAO VISUALIZAR CONFIRMAR DELETE USUARIO    
    $scope.visualizarConfirmarDelete = function (id) {
        $scope.idDelete = id;
        $scope.modalmessage = "Confirma exclusão do registro?";
        $("#modalConfirm").modal("show");
    };

    // BOTAO DELETAR USUARIO
    $scope.deletar = function (id) {
        factUsuarios.deletarUsuarios(id).then(function (d) {
            if (d.data.d.status == false) {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalConfirm").modal("hide");
                $("#modalDinamico").modal("show");
            }
            else {
                $rootScope.modalmessage = d.data.d.msg;
                if (!$scope.frmFiltroValues.Clogin) $scope.frmFiltroValues.Clogin = "";
                if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
                var query = Enumerable.From($rootScope.lstUsuarios);
                query = query.Where(function (x) { return x.Id === id }).Select(function (x) { return x; }).First();
                var idxobj = $rootScope.lstUsuarios.indexOf(query);
                $rootScope.lstUsuarios.splice(idxobj, 1);
                $("#modalConfirm").modal("hide");
                $("#modalDinamico").modal("show");
            }
        });
    };

    // BOTAO SUBMIT FILTRO
    $scope.submitFormFiltro = function () {
        if (!$scope.frmFiltroValues.Clogin) $scope.frmFiltroValues.Clogin = "";
        if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
        factUsuarios.obterUsuarios($scope.frmFiltroValues).then(function (d) {
            $rootScope.lstUsuarios = d.data.d;
            $scope.accordion.isFirstOpen = true;
        }, function (r) {
            alert(r);
        });
    };

    //BOTAO LIMPAR FILTRO
    $scope.limpaFormFiltro = function () {
        $scope.frmFiltroValues = {};
    };

    // BOTAO VISUALIZAR NOVO    
    $scope.visualizarNovo = function (objDados) {
        $scope.mfrmFiltroValues = {};
        $scope._tab_active = "usuarios";
        /*  variaveis de controle de validacao da tab modalusuario */
        $scope.isvalidtab1 = true;
        $scope.isvalidtab2 = true;
        $scope.isvalidtab3 = true;
        $scope.sel_usuario_privilegio = [];
        $scope.sel_usuario_conveniado = [];
        $scope.mfrmFiltroValues.Cativo = 'Ativo';
        $scope.mfrmFiltroValues.searchText = "";
        $("#modalUsuario").modal("show");
    };

    // BOTAO INSERIR/ATUALIZAR Usuário
    $scope.verificaacao = function () {
        if ($scope.mfrmFiltroValues.Clogin == undefined || $scope.mfrmFiltroValues.Clogin == '') {
            $scope.isvalidtab1 = false;
            $rootScope.modalmessage = 'Usuário é obrigatório!!!'
            $scope._tab_active = "usuarios";
            factFocus('mfrmFiltroValues.Clogin');

            return;
        }
        if ($scope.mfrmFiltroValues.Cativo == undefined) {
            $scope.isvalidtab1 = false;
            $rootScope.modalmessage = 'Status é obrigatório!!!'
            $scope._tab_active = "usuarios";    
            factFocus('mfrmFiltroValues.Cativo');

            return;
        }
        if ($scope.mfrmFiltroValues.Csenha == undefined || $scope.mfrmFiltroValues.Csenha == '') {
            $scope.isvalidtab1 = false;
            $rootScope.modalmessage = 'Senha é obrigatória!!!'
            $scope._tab_active = "usuarios";    
            factFocus('mfrmFiltroValues.Csenha');

            return;
        }

        if ($scope.mfrmFiltroValues.tb_usuario_privilegio == undefined || $scope.mfrmFiltroValues.tb_usuario_privilegio == "") {
            $rootScope.modalmessage = 'Privilégio é obrigatório!!!'
            $scope.isvalidtab2 = false;
            $scope._tab_active = "privilegios";    
  
            return;
        }

        if ($scope.mfrmFiltroValues.tb_conveniado_usuario == undefined || $scope.mfrmFiltroValues.tb_conveniado_usuario == "") {
            $rootScope.modalmessage = 'Conveniado é obrigatório!!!'
            $scope.isvalidtab3 = false;
            $scope._tab_active = "conveniados";
            factFocus('selconveniado');
            return;
        }

        if (!$scope.mfrmFiltroValues.Id) $scope.mfrmFiltroValues.Id = 0;
        factUsuarios.definirUsuarios($scope.mfrmFiltroValues).then(function (d) {
            if (d.data.d.status === false) {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalUsuario").modal("hide");
                $("#modalDinamico").modal("show");
            }
            else {
                var retUsuario = d.data.d;
                $rootScope.modalmessage = d.data.d.msg;
                if (!$scope.frmFiltroValues.Clogin) $scope.frmFiltroValues.Clogin = "";
                if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
                factUsuarios.obterUsuarios($scope.frmFiltroValues).then(function (d) {
                    $rootScope.lstUsuarios = d.data.d;
                    $scope.accordion.isFirstOpen = true;
                });
                if ($rootScope.idLogin == $scope.mfrmFiltroValues.Id) factLogin.definirSession(retUsuario).then(function (d) {
                    if (d.data.d.status === true) {
                        $rootScope.idLogin = d.data.d.IdLogin;
                        $rootScope.userPrivilegio = d.data.d.CsccPrivilegio;
                        $rootScope.userConveniado = d.data.d.CsccConveniado;
                        factConveniado.initConveniado();
                    }
                });
                $("#modalUsuario").modal("hide");
                $("#modalDinamico").modal("show");
            }

    
        });

    }


    $scope.adicionaNovoConveniadoFiltrado = function (lista, filtro) {
        $rootScope.modalmessage = 'Ja cadastrados: '
        for (i = 0 ; i < lista.length; i++)
        {
            var sel = lista[i];

            var ipos = $scope.sel_usuario_conveniado.arrayObjectIndexOf(sel.Id, 'Id_conveniado');
            if (ipos < 0) {

                if (sel.Cnome.toLowerCase().indexOf(filtro.toLowerCase()) >= 0)
                {
                    // INSERE O CONVENIADO NO ARRAY LOCAL
                    $scope.sel_usuario_conveniado.push({ ID: 0, Id_conveniado: sel.Id, Id_usuario: $scope.mfrmFiltroValues.Id, tb_conveniado:sel });
                    // se nao existe nenhum usuario conveniado criado cria o relacionamento vazio
                    if ($scope.mfrmFiltroValues.tb_conveniado_usuario == undefined || $scope.mfrmFiltroValues.tb_conveniado_usuario == null) {
                        $scope.mfrmFiltroValues.tb_conveniado_usuario = [];
                    }
                    $scope.mfrmFiltroValues.tb_conveniado_usuario.push({ Id: 0, Id_conveniado: sel.Id, Id_usuario: $scope.mfrmFiltroValues.Id, msg: 'novo' });
                }
            } else {
                $rootScope.modalmessage += "" + sel.Cnome + ";"
                $("#modalDinamico").modal("show");
            }

        }

        // alert(1);

    }


    // ADICIONAR UM CONVENIADO
    $scope.adicionaNovoConveniado = function (selected) {
        if (selected == "novo") {
            factConveniado.clearModalConveniado();
            $("#modalConveniado").modal("show");
        }
        else {
            //Verifica se o conveniado ja existe no array local 
            //var ipos = arrayObjectIndexOf($scope.sel_usuario_conveniado, JSON.parse(selected).Id, 'Id_conveniado');
            var ipos = $scope.sel_usuario_conveniado.arrayObjectIndexOf(JSON.parse(selected).Id, 'Id_conveniado');
            if (ipos < 0) {
                // INSERE O CONVENIADO NO ARRAY LOCAL
                $scope.sel_usuario_conveniado.push({ ID: 0, Id_conveniado: JSON.parse(selected).Id, Id_usuario: $scope.mfrmFiltroValues.Id, tb_conveniado: JSON.parse(selected) });
                // se nao existe nenhum usuario conveniado criado cria o relacionamento vazio
                if ($scope.mfrmFiltroValues.tb_conveniado_usuario == undefined || $scope.mfrmFiltroValues.tb_conveniado_usuario == null) {
                    $scope.mfrmFiltroValues.tb_conveniado_usuario = [];
                }
                $scope.mfrmFiltroValues.tb_conveniado_usuario.push({ Id: 0, Id_conveniado: JSON.parse(selected).Id, Id_usuario: $scope.mfrmFiltroValues.Id, msg: 'novo' });
            } else {
                $rootScope.modalmessage = "Conveniado: " + JSON.parse(selected).Cnome + " já está cadastrado!"
                $("#modalDinamico").modal("show");
            }
        }
    };

    // APAGAR UM CONVENIADO
    $scope.deletarConveniado = function (objConveniado) {
        // REMOVE O CONVENIADO DO ARRAY LOCAL
        for (var i = $scope.sel_usuario_conveniado.length - 1; i >= 0; i--) {
            if ($scope.sel_usuario_conveniado[i].Id_conveniado === objConveniado.Id) {
                $scope.sel_usuario_conveniado.splice(i, 1);
                break;
            }
        }
        // AGORA REMOVE DO frmMODAL PARA SALVAR
        for (var i = 0 ; i < $scope.mfrmFiltroValues.tb_conveniado_usuario.length ; i++) {
            if ($scope.mfrmFiltroValues.tb_conveniado_usuario[i].Id_conveniado == objConveniado.Id) {
                $scope.mfrmFiltroValues.tb_conveniado_usuario[i].msg = 'deletado';
                break;
            }
        }
    };

    // Retorna TRUE se o usuario tem o privilegio ou FALSE se nao.
    $scope.obter_check = function (Id) {
        var ch = false;
        for (var i = 0 ; i < $scope.sel_usuario_privilegio.length ; i++) {
            if (Id == $scope.sel_usuario_privilegio[i].Id_privilegio) {
                ch = true;
                break;
            }
        }
        return ch;
    }

    // clicou no check box de privilegio
    $scope.clicou_check = function (Privilegio) {
        //var chk = $scope.obter_check(Privilegio.Id);
        //Se nao existe nenhum usuario privilegio criado cria o relacionamento vazio
        if ($scope.mfrmFiltroValues.tb_usuario_privilegio == undefined || $scope.mfrmFiltroValues.tb_usuario_privilegio == null) $scope.mfrmFiltroValues.tb_usuario_privilegio = [];

        var jaexiste = false;
        //Procura pelo privilegios do usuario
        for (var i = 0 ; i < $scope.mfrmFiltroValues.tb_usuario_privilegio.length ; i++) {
            if ($scope.mfrmFiltroValues.tb_usuario_privilegio[i].Id_privilegio == Privilegio.Id) {
                jaexiste = true;
                // se ja existia na tabela como NULL entao manda DELETAR  
                if ($scope.mfrmFiltroValues.tb_usuario_privilegio[i].msg == null) {
                    $scope.mfrmFiltroValues.tb_usuario_privilegio[i].msg = 'deletado';
                    break;
                } else if ($scope.mfrmFiltroValues.tb_usuario_privilegio[i].msg == 'novo') {
                    $scope.mfrmFiltroValues.tb_usuario_privilegio.splice(Privilegio.Id, 1);
                }
                else {
                    // se estava como DELETADO volta para NULL
                    $scope.mfrmFiltroValues.tb_usuario_privilegio[i].msg = null;
                    break;
                }
            }
        }
        if (!jaexiste) {
            // se nao existe SETA o ID para 0 e faz um PUSH no objeto
            if ($scope.mfrmFiltroValues.Id == "") $scope.mfrmFiltroValues.Id = 0;
            $scope.mfrmFiltroValues.tb_usuario_privilegio.push({ Id: 0, Id_privilegio: Privilegio.Id, Id_usuario: $scope.mfrmFiltroValues.Id, msg: 'novo', tb_privilegio: Privilegio });
        }
    }

    _initUsuario();
    $scope.initUsuario = _initUsuario; //Aqui definimos o que será exposto no escopo
}]);