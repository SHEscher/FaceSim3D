app.directive('eventFocus', function (focus) {
    return function (scope, elem, attr) {
        elem.on(attr.eventFocus, function () {
            focus(attr.eventFocusId)
        });

        scope.$on('$destroy', function (d) {
            element.off(attr.eventFocus);
        });

    };
})

