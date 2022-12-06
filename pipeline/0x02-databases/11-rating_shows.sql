-- Script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.
SELECT
    tv_shows.name AS genre,
    count(tv_show_genres.show_id) AS number_of_shows
FROM
    tv_shows
    INNER JOIN tv_show_genres ON tv_genres.id = tv_show_genres.genre_id
GROUP BY
    tv_shows.name
ORDER BY
    number_of_shows DESC;