from django.contrib.staticfiles.storage import staticfiles_storage
from django.urls import reverse
from jinja2 import Environment
from django.utils.translation import gettext, ngettext


def jinja2_environment(**options):
    env = Environment(**options, extensions=['jinja2.ext.i18n'])
    env.install_gettext_callables(gettext=gettext, ngettext=ngettext, newstyle=True)
    env.globals.update({
        'static': staticfiles_storage.url,
        'url': reverse,
    })
    return env

