app.factory('factFocus', function ($timeout) {
    return function (id) {
        $timeout(function () {

            var element = document.getElementById(id);
            if (element)
                element.focus();
        })
    }
});