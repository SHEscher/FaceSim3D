app.controller("ContratoCtrl", ['$scope', '$rootScope', 'factContrato', 'factCredenciado', 'factConveniado', 'factEmpresa', '$element',"$timeout", function ($scope, $rootScope, factContrato, factCredenciado, factConveniado, factEmpresa, $element,$timeout) {
    function _initContrato() {
        /* Aqui vai tudo que deve ser inicializado */
        $scope.frmFiltroValues = { tb_credenciado : {} };

        /* Variáveis de controle */
        $scope.accordion = {
            isFirstOpen: true,
            isFilterOpen: false
        };

        $scope.isvalid = true;

   
    }


    //$scope.teste = function () {
    //    var _elm = $("#modalContrato");
    //    var pick1 = _elm.find('.selectpickeremp');
    //    pick1.selectpicker('render');
    //    pick1.selectpicker('refresh');
    //}

    //WATCH
    var watchPicker = $scope.$watch("frmModalContrato.Ccategoria", function (nv, ov) {
        console.log("nv");
        $timeout(function () {
            var _elm = $("#modalContrato");
            var pick1 = _elm.find('.selectpickeremp');
            pick1.selectpicker('render');
            pick1.selectpicker('refresh');
           /// $scope.$apply();
        }, 1000);
    });

    $scope.$on("$destroy", function () {
        watchPicker(); //Destroi watch
    });

    //
    //BOTAO VISUALIZAR
    $scope.visualizar = function (objDados) {
        $scope.selecttab("Credenciado");
        $scope.frmModalContrato = JSON.parse(JSON.stringify(objDados));

        $scope.frmModalContrato.Tsvigencia_inicio = moment($scope.frmModalContrato.Tsvigencia_inicio);
        $scope.frmModalContrato.Tsvigencia_fim = moment($scope.frmModalContrato.Tsvigencia_fim);

        $scope.$apply();

        $("#modalContrato").modal("show");

        $rootScope.modalmessage = '';

        var _elm = $("#modalContrato");
       
        // Picker de categoria
        var pick = _elm.find('.selectpickercat');
        pick.selectpicker();
        var selectedPickCat = [];
        for (var i in objDados.tb_contrato_conveniado)
            objDados.tb_contrato_conveniado[i].Id_categoria ? selectedPickCat.push(objDados.tb_contrato_conveniado[i].Id_categoria) : i;

        pick.selectpicker('val', selectedPickCat);
        pick.selectpicker('render');
        pick.selectpicker('refresh');
        $scope.frmModalContrato.Ccategoria = pick.selectpicker('val');
        $scope.$apply();


        // Picker de empresa
        var pick1 = _elm.find('.selectpickeremp');
        pick1.selectpicker();
        var selectedPickEmp =[];
        for (var i in objDados.tb_contrato_conveniado)
            objDados.tb_contrato_conveniado[i].id_empresa ? selectedPickEmp.push(objDados.tb_contrato_conveniado[i].id_empresa): i;

        pick1.selectpicker('val', selectedPickEmp);
        pick1.selectpicker('render');
        pick1.selectpicker('refresh');
        $scope.frmModalContrato.Cempresa = pick1.selectpicker('val');

    };

    //BOTAO VISUALIZAR NOVO
    $scope.visualizarNovo = function (objDados) {
        factCredenciado.clearConfirmCredenciado();
        $rootScope.modalmessage = '';
        $("#modalConfirmCredenciado").modal("show");
    };

    //BOTAO VISUALIZAR CONFIRMAR DELETE
    $scope.visualizarConfirmarDelete = function (id) {
        $scope.idDelete = id;
        $scope.modalmessage = "Confirma exclusão do registro?";
        $("#modalConfirm").modal("show");
    };

    //BOTAO DELETAR
    $scope.deletar = function (id) {
        factContrato.deletarContrato(id).then(function (d) {
            if (d.data.d.status == false) {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalConfirm").modal("hide");
                $("#modalDinamico").modal("show");
            }
            else {
                $rootScope.modalmessage = d.data.d.msg;
                var query = Enumerable.From($rootScope.lstContrato);
                query = query.Where(function (x) { return x.Id === id }).Select(function (x) { return x; }).First();
                var idxobj = $rootScope.lstContrato.indexOf(query);
                $rootScope.lstContrato.splice(idxobj, 1);
                $("#modalConfirm").modal("hide");
                $("#modalDinamico").modal("show");

            }
        });
    };

    //BOTAO SUBMIT FILTRO
    $scope.submitFormFiltro = function () {
        if (!$scope.frmFiltroValues.tb_credenciado.Cnome) $scope.frmFiltroValues.tb_credenciado.Cnome = "";
        if (!$scope.frmFiltroValues.Cnro_contrato) $scope.frmFiltroValues.Cnro_contrato = "";
        if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
        factContrato.obterContrato($scope.frmFiltroValues).then(function (d) {
            $rootScope.lstContrato = d.data.d;
            $scope.accordion.isFirstOpen = true;
        });
    };

    //BOTAO LIMPAR FILTRO
    $scope.limpaFormFiltro = function () {
        $scope.frmFiltroValues = {};
    };

    //BOTAO DELETAR TabVAICULOS (ModalContrato)
    $scope.deletarVeiculoContrato = function (objDados) {
        objDados.msg = 'deleted';
    };

    //BOTAO ADICIONAR TabVEICULOS (ModalContrato)
    $scope.adicionarNovoVeiculo = function () {
        if (!$scope.frmModalContrato.tb_usertag) $scope.frmModalContrato.tb_usertag = [];

        $scope.frmModalContrato.CplacaAdd = $scope.frmModalContrato.CplacaAdd.toUpperCase();


        var listIndex = $scope.frmModalContrato.tb_usertag.arrayObjectIndexOf($scope.frmModalContrato.CplacaAdd, 'Cplaca');
        if (listIndex >= 0) {
            $scope.frmModalContrato.tb_usertag[listIndex].msg = '';
        }
        else {
            var IIssuerAdd = $scope.frmModalContrato.CtagAdd.substr(0, 5);
            var LTagAdd = $scope.frmModalContrato.CtagAdd.substr(6, 10);
            $scope.frmModalContrato.tb_usertag.push({ Cplaca: $scope.frmModalContrato.CplacaAdd, Ltag: LTagAdd, IIssuer: IIssuerAdd, Id_contrato: $scope.frmModalContrato.Id, Cdescricao: $scope.frmModalContrato.tb_credenciado.Cnome });
        }
        $scope.frmModalContrato.CplacaAdd = '';
        $scope.frmModalContrato.CtagAdd = '';
    };

    //VALIDA PLACA (ConfirmCredenciado)
    $scope.validarPlacaTAGconfirm = function (_Cplaca) {
        factCredenciado.validarPlacaTAG(_Cplaca).then(function (d) {
            var IIssuer = d.data.d.IIssuer;
            var Ctag = d.data.d.LTag;
            d.data.d.status == false ? $scope.modalConfirmCredenciado.Ctag = d.data.d.msg : $scope.modalConfirmCredenciado.Ctag = IIssuer.toString().preencherEsq(0, 5) + Ctag.toString().preencherEsq(0, 10);
        });
    };

    //VALIDA PLACA (ModalContrato)
    $scope.validarPlacaTAGmodal = function (_Cplaca) {
        factCredenciado.validarPlacaTAG(_Cplaca).then(function (d) {
            var IIssuer = d.data.d.IIssuer;
            var Ctag = d.data.d.LTag;
            d.data.d.status == false ? $scope.frmModalContrato.CtagAdd = d.data.d.msg : $scope.frmModalContrato.CtagAdd = IIssuer.toString().preencherEsq(0, 5) + Ctag.toString().preencherEsq(0, 10);
        });
    };

    //BOTAO SUBMIT (ConfirmCredenciado)
    $scope.submitConfirmCredenciado = function () {
        factCredenciado.validarCredenciado($scope.modalConfirmCredenciado).then(function (d) {
            if (d.data.d.status == false) {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalDinamico").modal("show");
            }
            else {
                //Verifica CNPJ/CPF com cadastro
                if (d.data.d.numeroDocumento != $scope.modalConfirmCredenciado.Ccnpj_cpf) {
                    $scope.modalConfirmCredenciado.Ccnpj_cpf.length > 11 ? $rootScope.modalmessage = "CNPJ " : $rootScope.modalmessage = "CPF ";
                    $rootScope.modalmessage += "informado não correspondente para a Placa " + $scope.modalConfirmCredenciado.Cplaca.toUpperCase();
                    $("#modalDinamico").modal("show");
                }
                else {
                    //Verifica Credenciado
                    var CnumeroCelular = d.data.d.numeroCelular;
                    $scope.modalConfirmCredenciado.Ctag = '00290' + d.data.d.numeroTag;
                    factCredenciado.obterCredenciado($scope.modalConfirmCredenciado).then(function (d) {
                        /*Credenciado NAO cadastrado*/
                        if (d.data.d[0].status == false) {
                            var frmModalCredenciado = {};
                            frmModalCredenciado = $scope.modalConfirmCredenciado;
                            frmModalCredenciado.Cfonecel = CnumeroCelular;
                            $("#modalConfirmCredenciado").modal("hide");
                            $scope.frmModalCredenciado = JSON.parse(JSON.stringify(frmModalCredenciado));;
                            frmModalContrato.Cativo = 'Ativo';
                            $("#modalCredenciado").modal("show");
                        }
                            /*Credenciado JA cadastrado*/
                        else {
                            var frmModalContrato = {};
                            $("#modalConfirmCredenciado").modal("hide");
                            factContrato.clearModalContrato();
                            frmModalContrato.tb_credenciado = d.data.d[0];

                            if (!frmModalContrato.tb_usertag) frmModalContrato.tb_usertag = [];
                            var _placa = $scope.modalConfirmCredenciado.Cplaca.toUpperCase();
                            // var _placa = $scope.modalConfirmCredenciado.Cplaca;
                            var _iissuer = $scope.modalConfirmCredenciado.Ctag.substr(0, 5);
                            var _ltag = $scope.modalConfirmCredenciado.Ctag.substr(6, 10);
                            var _nome = frmModalContrato.tb_credenciado.Cnome;

                           

                            frmModalContrato.tb_usertag.push({ Cplaca: _placa, Ltag: _ltag, IIssuer: _iissuer, Id_contrato: 0, Cdescricao: _nome });
                           // frmModalContrato.Cativo = 'Ativo';

                            $scope.visualizar(frmModalContrato);
                        }
                    });
                }
            }
        });
    };

    //BOTAO SUBMIT (ModalCredenciado)
    $scope.submitModalCredenciado = function () {
        if (!$scope.frmModalCredenciado.Id) $scope.frmModalCredenciado.Id = 0;
        if (!$scope.frmModalCredenciado.IIssuer) $scope.frmModalCredenciado.IIssuer = $scope.frmModalCredenciado.Ctag.substr(0, 5);
        if (!$scope.frmModalCredenciado.LTag) $scope.frmModalCredenciado.LTag = $scope.frmModalCredenciado.Ctag.substr(6, 10);
        if (!$scope.frmModalCredenciado.Id_empresa) $scope.frmModalCredenciado.Id_empresa = $scope.frmModalCredenciado.Cempresa;
        if (!$scope.frmModalCredenciado.Csenha) $scope.frmModalCredenciado.Csenha = $scope.frmModalCredenciado.Ccnpj_cpf.substr(0, 5);
        $scope.frmModalCredenciado.Ccnpj_cpf.length > 11 ? $scope.frmModalCredenciado.Ctipo_documento = "CNPJ" : $scope.frmModalCredenciado.Ctipo_documento = "CPF";
        factCredenciado.definirCredenciado($scope.frmModalCredenciado).then(function (d) {
            if (d.data.d.status == false) {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalCredenciado").modal("hide");
                $("#modalDinamico").modal("show");
            }
            else {
                $rootScope.modalmessage = d.data.d.msg;
                if ($scope.frmModalCredenciado.Id != 0) {
                    //DELETA ITEM ANTERIOR
                    var query = Enumerable.From($rootScope.lstCredenciado);
                    query = query.Where(function (x) { return x.Id === $scope.frmModalCredenciado.Id }).Select(function (x) { return x; }).First();
                    var idxobj = $rootScope.lstCredenciado.indexOf(query);
                    $rootScope.lstCredenciado.splice(idxobj, 1);
                }
                $rootScope.lstCredenciado.push(d.data.d);
                $("#modalCredenciado").modal("hide");
                var frmModalContrato = {};
                frmModalContrato.tb_credenciado = $scope.frmModalCredenciado;
                factContrato.clearModalContrato();
                $scope.visualizar(frmModalContrato);
            }
        });
    };

    //BOTAO SUBMIT (ModalContrato)
    $scope.submitModalContrato = function () {
        $scope.isvalid = true;

        //var _elm = $("#modalContrato");



        if ($scope.frmModalContrato.Ccategoria == null || $scope.frmModalContrato.Ccategoria == undefined || $scope.frmModalContrato.Ccategoria.length == 0) {
            $scope.isvalid = false;
            $rootScope.modalmessage = 'Categoria é obrigatória!!!';
            $scope.selecttab("Contrato");
            $rootScope.$apply();
            return;
        }

        if ($scope.frmModalContrato.Cempresa == null || $scope.frmModalContrato.Cempresa == undefined || $scope.frmModalContrato.Cempresa.length == 0) {
            $scope.isvalid = false;
            $rootScope.modalmessage = 'Empresa é obrigatória!!!';
            $scope.selecttab("Contrato");
            $rootScope.$apply();
            return;
        }

        if ( $scope.frmModalContrato.Ccategoria.length  != $scope.frmModalContrato.Cempresa.length )  {
            $scope.isvalid = false;
            $rootScope.modalmessage = 'Selecione mais ' + ($scope.frmModalContrato.Ccategoria.length - $scope.frmModalContrato.Cempresa.length) + ' empresas.';
            $scope.selecttab("Contrato");
            $rootScope.$apply();
            return;
        }



        if ($scope.frmModalContrato.Cnro_contrato == null || $scope.frmModalContrato.Cnro_contrato == undefined || $scope.frmModalContrato.Cnro_contrato.length == 0) {
            $scope.isvalid = false;
            $rootScope.modalmessage = 'Identificação do Contrato é obrigatória!!!';
            $scope.selecttab("Contrato");
            $rootScope.$apply();
            return;
        }
        if ($scope.frmModalContrato.Iqtde_vagas == null || $scope.frmModalContrato.Iqtde_vagas == undefined || $scope.frmModalContrato.Iqtde_vagas.length == 0) {
            $scope.isvalid = false;
            $rootScope.modalmessage = 'Quantidade de vagas é obrigatória!!!';
            $scope.selecttab("Contrato");
            $rootScope.$apply();
            return;
        }

        if ($scope.frmModalContrato.Ivalor == null || $scope.frmModalContrato.Ivalor == undefined || $scope.frmModalContrato.Ivalor.length == 0) {
            $scope.isvalid = false;
            $rootScope.modalmessage = 'Valor do Contrato é obrigatório!!!';
            $scope.selecttab("Contrato");
            $rootScope.$apply();
            return;
        }

        if ($scope.frmModalContrato.Tsvigencia_inicio == null || $scope.frmModalContrato.Tsvigencia_inicio == undefined || $scope.frmModalContrato.Tsvigencia_inicio.length == 0) {
            $scope.isvalid = false;
            $rootScope.modalmessage = 'Vigência Início é obrigatória!!!';
            $scope.selecttab("Contrato");
            $rootScope.$apply();
            return;
        }

        if ($scope.frmModalContrato.Tsvigencia_fim == null || $scope.frmModalContrato.Tsvigencia_fim == undefined || $scope.frmModalContrato.Tsvigencia_fim.length == 0) {
            $scope.isvalid = false;
            $rootScope.modalmessage = 'Vigência Fim é obrigatória!!!';
            $scope.selecttab("Contrato");
            $rootScope.$apply();
            return;
        }
        if ($scope.frmModalContrato.Cativo == null || $scope.frmModalContrato.Cativo == undefined || $scope.frmModalContrato.Cativo.length == 0) {
            $scope.isvalid = false;
            $rootScope.modalmessage = 'Status é obrigatório!!!';
            $scope.selecttab("Contrato");
            $rootScope.$apply();
            return;
        }




        if (!$scope.frmModalContrato.Id) $scope.frmModalContrato.Id = 0;
        if (!$scope.frmModalContrato.Id_credenciado) $scope.frmModalContrato.Id_credenciado = $scope.frmModalContrato.tb_credenciado.Id;
        if (!$scope.frmModalContrato.Ccategoria) $scope.frmModalContrato.Ccategoria = [];
        if (!$scope.frmModalContrato.tb_contrato_conveniado) $scope.frmModalContrato.tb_contrato_conveniado = [];
  

        // Varre pelas categorias, adiciona quem nào está lá e coloca msg=modified caso esteja
        $scope.frmModalContrato.Ccategoria.forEach(function (cat) {
            var query = Enumerable.From($scope.frmModalContrato.tb_contrato_conveniado);
            query = query.Where(function (cc) { return cc.Id_categoria == cat }).Select(function (categoria) { return categoria; }).FirstOrDefault(null);

            if (!query) {
                ///precisa adicionar
                ///após adicionad query.msg='added';
                var idconv = $scope.dicionarioCategorias[cat];
                // procura pelo id_empresa em dicionario empresa

                for (i = 0; i < $scope.frmModalContrato.Cempresa.length; i++)
                {
                    var idEmp = $scope.frmModalContrato.Cempresa[i];
                    var idConvEmp = $scope.dicionarioEmpresas[idEmp];
                    if (idconv==idConvEmp)
                    {
                        var idEmpresa = idEmp;
                    }

                }

                if (idEmpresa == null || idEmpresa == undefined) {
                    $scope.isvalid = false;
                    $rootScope.modalmessage = 'Empresa é obrigatória!!!';
                    return;

                }

                $scope.frmModalContrato.tb_contrato_conveniado.push({ Id: 0, Id_conveniado: idconv, Id_categoria: cat, Id_contrato: $scope.frmModalContrato.Id, Id_usuario: $rootScope.idLogin, msg: 'added', Id_empresa: idEmpresa });
            } else {

                var idconv = $scope.dicionarioCategorias[cat];
                // procura pelo id_empresa em dicionario empresa

                for (i = 0; i < $scope.frmModalContrato.Cempresa.length; i++) {
                    var idEmp = $scope.frmModalContrato.Cempresa[i];
                    var idConvEmp = $scope.dicionarioEmpresas[idEmp];
                    if (idconv == idConvEmp) {
                        var idEmpresa = idEmp;
                    }

                }

                query.id_empresa = idEmpresa;

                query.Id_usuario = $rootScope.idLogin;
                query.msg = "modified";
            };

        });


        // Procura por TODO: mundo que nào possui MSG
        $scope.frmModalContrato.tb_contrato_conveniado.forEach(function (_c_toremove) {
            if (!_c_toremove.msg) {
                _c_toremove.msg = "deleted";
            };
        });
      //  $scope.frmModalContrato.Ivalor = 2;
        var objDados = JSON.parse(JSON.stringify($scope.frmModalContrato));
        factContrato.definirContrato(objDados).then(function (d) {
            if (d.data.d.status == false) {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalContrato").modal("hide");
                $("#modalDinamico").modal("show");
            }
            else {
                $rootScope.modalmessage = d.data.d.msg;
                if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = "";
                if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
                if ($scope.frmModalContrato.Id != 0) {
                    //DELETA ITEM ANTERIOR
                    var query = Enumerable.From($rootScope.lstContrato);
                    query = query.Where(function (x) { return x.Id === $scope.frmModalContrato.Id }).Select(function (x) { return x; }).First();
                    var idxobj = $rootScope.lstContrato.indexOf(query);
                    $rootScope.lstContrato.splice(idxobj, 1);
                }
                d.data.d.Ivalor = d.data.d.Ivalor / 100;
                $rootScope.lstContrato.push(d.data.d);
                $("#modalContrato").modal("hide");
                $("#modalDinamico").modal("show");
            }
        }, function (d) {
            console.log(d)
        }
        );
    };

    $scope.tttcat = function (categoria) {
        if (!$scope.indiceCategorias) {
            $scope.indiceCategorias = [];
        };

        $scope.indiceCategorias.push(
            {
                "idcategoria": categoria.Id,
                "idconveniado": categoria.Id_conveniado
            }
            );

        if (!$scope.dicionarioCategorias) {
            $scope.dicionarioCategorias = {};
        }
        $scope.dicionarioCategorias[categoria.Id] = categoria.Id_conveniado;
    }

    $scope.tttemp = function (empresa) {
        if (!$scope.indiceEmpresas) {
            $scope.indiceEmpresas = [];
        };

        $scope.indiceEmpresas.push(
            {
                "idempresa": empresa.Id,
                "idconveniado": empresa.Id_conveniado
            }
            );

        if (!$scope.dicionarioEmpresas) {
            $scope.dicionarioEmpresas = {};
        }
        $scope.dicionarioEmpresas[empresa.Id] = empresa.Id_conveniado;
    }

    // SETA TAB SELECIONADA
    $scope.selecttab = function (settab) {
        $scope._tab_active = settab;
        $scope.$apply();
    };

    $scope.selecttab("Credenciado");
    _initContrato();
    $scope.initContrato = _initContrato; //Aqui definimos o que será exposto no escopo
}]);